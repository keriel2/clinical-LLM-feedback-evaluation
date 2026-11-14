"""
compute_profiler.py
-------------------
Compute-cost accounting for FMACA-CE revision (reviewer item: report critic cost
and training overhead).

Wraps trainer and component methods at instance level, so no training logic changes.
Records, per run:
  - wall-clock seconds and call counts per component
      sft_step, contrastive_step, generation, critic, selection, feedback, adapter
  - hardware-independent counts: supervised optimiser steps, contrastive steps,
    contrastive forward passes, generated outputs
  - peak GPU memory per visible device
Times are measured with torch.cuda.synchronize() so GPU work is attributed correctly.

Writes <condition_dir>/compute_profile.json
"""

import os
import json
import time
import socket
from collections import defaultdict


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
    except Exception:
        pass


class ComputeProfiler:
    def __init__(self):
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.items = defaultdict(int)       # e.g. prompts generated, examples scored
        self._t0 = time.perf_counter()
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _wrap(self, obj, method, key, count_arg=None, gpu=True):
        if obj is None or not hasattr(obj, method):
            return
        orig = getattr(obj, method)
        prof = self

        def wrapped(*args, **kwargs):
            if gpu:
                _sync()
            t0 = time.perf_counter()
            result = orig(*args, **kwargs)
            if gpu:
                _sync()
            prof.seconds[key] += time.perf_counter() - t0
            prof.calls[key] += 1
            if count_arg is not None:
                seq = kwargs.get(count_arg[1]) if count_arg[1] in kwargs else (
                    args[count_arg[0]] if len(args) > count_arg[0] else None)
                if seq is not None:
                    try:
                        prof.items[key] += len(seq)
                    except TypeError:
                        pass
            return result

        setattr(obj, method, wrapped)

    def attach_trainer(self, trainer):
        # train_step_only includes the contrastive step when it fires; both are timed,
        # and sft-only time is derived as the difference.
        self._wrap(trainer, "train_step_only", "train_step_total", count_arg=(0, "examples"))
        self._wrap(trainer, "_contrastive_step", "contrastive_step")
        self._wrap(trainer, "generate_outputs", "generation", count_arg=(0, "prompts"))
        return self

    def attach_coordinator(self, coordinator):
        # Components run on CPU: no GPU sync needed
        self._wrap(coordinator.evaluation_agent, "evaluate_batch", "critic",
                   count_arg=(0, "outputs"), gpu=False)
        self._wrap(coordinator.selection_agent, "select", "selection", gpu=False)
        self._wrap(coordinator.feedback_agent, "process", "feedback", gpu=False)
        self._wrap(coordinator.adapter_agent, "select", "adapter", gpu=False)
        return self

    # ------------------------------------------------------------------
    def dump(self, path, condition, rounds, batch_size, extra=None):
        _sync()
        wall = time.perf_counter() - self._t0
        sec = dict(self.seconds)
        contrastive = sec.get("contrastive_step", 0.0)
        sft_only = sec.get("train_step_total", 0.0) - contrastive
        component_keys = ["generation", "critic", "selection", "feedback", "adapter"]
        accounted = sft_only + contrastive + sum(sec.get(k, 0.0) for k in component_keys)

        gpus = []
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpus.append({
                        "index_visible": i,
                        "name": torch.cuda.get_device_name(i),
                        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(i) / 2**30, 2),
                        "peak_reserved_gb": round(torch.cuda.max_memory_reserved(i) / 2**30, 2),
                    })
        except Exception:
            pass

        n_contrastive = self.calls.get("contrastive_step", 0)
        profile = {
            "condition": condition,
            "rounds": rounds,
            "batch_size": batch_size,
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "n_gpus": len(gpus),
            "gpus": gpus,
            "wall_seconds": round(wall, 1),
            "seconds": {
                "sft_only": round(sft_only, 2),
                "contrastive_step": round(contrastive, 2),
                **{k: round(sec.get(k, 0.0), 2) for k in component_keys},
                "other": round(wall - accounted, 2),   # model loading, saving, Python overhead
            },
            "counts": {
                "supervised_optimizer_steps": self.calls.get("train_step_total", 0),
                "supervised_examples": self.items.get("train_step_total", 0),
                "contrastive_steps": n_contrastive,
                "contrastive_forward_passes_max": n_contrastive * 4 * 2,  # 4 pairs x (wrong, chosen)
                "generation_calls": self.calls.get("generation", 0),
                "generated_outputs": self.items.get("generation", 0),
                "critic_calls": self.calls.get("critic", 0),
                "critic_outputs_scored": self.items.get("critic", 0),
            },
            "per_round_ms": {
                k: round(1000 * v / max(rounds, 1), 2)
                for k, v in {"sft_only": sft_only, "contrastive_step": contrastive,
                             **{c: sec.get(c, 0.0) for c in component_keys}}.items()
            },
        }
        if extra:
            profile.update(extra)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)
        return profile
