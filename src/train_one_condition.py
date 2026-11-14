"""
train_one_condition.py
----------------------
DeepSpeed-compatible training script for one FMACA condition.
Launch with: deepspeed --include localhost:6,7 train_one_condition.py --condition fmaca
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, '.')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train_one")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",
                        default="${MODELS_DIR}/Qwen2.5-7B-Instruct")
    parser.add_argument("--condition",       default="fmaca",
                        choices=["static_sft","quality_only","error_curriculum",
                                 "dynamic_adapter","fmaca","fmaca_nocontrastive","static_coord","odm"])
    parser.add_argument("--output_dir",      default="./ablation_v2")
    parser.add_argument("--epochs",          type=int, default=3)
    parser.add_argument("--batch_size",      type=int, default=8)
    parser.add_argument("--max_per_dataset", type=int, default=1000)
    parser.add_argument("--datasets",        nargs="+", default=["medqa"])
    parser.add_argument("--local_rank",      type=int, default=0)
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    is_main    = (local_rank == 0)

    # Only initialise distributed if launched with deepspeed/torchrun
    # Single GPU mode: skip distributed init entirely
    use_distributed = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", 1)) > 1
    if use_distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    if is_main:
        n_gpus = dist.get_world_size() if use_distributed else 1
        logger.info("Starting condition: %s on %d GPU(s)", args.condition, n_gpus)

    # --- Load pre-sampled fixed data (same for ALL conditions) ---
    # This ensures fair comparison and reproducibility
    shared_data_path = "./shared_train_data.json"
    if os.path.exists(shared_data_path):
        import json
        with open(shared_data_path) as f:
            train_data = json.load(f)
        if is_main:
            logger.info(
                "Loaded %d pre-sampled examples from %s",
                len(train_data), shared_data_path,
            )
    else:
        # Fallback: load fresh (less reproducible)
        if is_main:
            logger.warning(
                "shared_train_data.json not found — run prepare_data.py first"
            )
        from data_loader import FMACADataLoader
        loader = FMACADataLoader()
        train_data = loader.load_all(
            split="train",
            datasets=args.datasets,
            max_per_dataset=args.max_per_dataset,
        )
        train_data = [d for d in train_data if d.get("answer","?") != "?"]
        if is_main:
            logger.info("Loaded %d examples", len(train_data))

    # --- Load trainer (each rank loads its own model shard) ---
    from trainer import FMACATrainer
    use_contrastive = (args.condition == "fmaca")
    trainer = FMACATrainer(
        model_path=args.model_path,
        use_deepspeed=False,
        use_contrastive=use_contrastive,
    )
    if use_contrastive:
        trainer._contrastive_enabled = True
        logger.info("Contrastive signal ENABLED for FMACA")
    trainer.set_adapter("oft")
    if is_main:
        logger.info("Trainer ready | adapter=%s", trainer._current_mode)

    # Revision: compute-cost accounting (timing wrappers only; training unchanged)
    from compute_profiler import ComputeProfiler
    profiler = ComputeProfiler().attach_trainer(trainer)
    # BENCH_ROUNDS caps the run length for the controlled compute benchmark (0 = full run)
    bench_rounds = int(os.environ.get("BENCH_ROUNDS", "0"))

    # ------------------------------------------------------------------
    # STATIC SFT BASELINE: bypass Coordinator/agents completely
    # ------------------------------------------------------------------
    # static_sft should be a plain supervised fine-tuning baseline:
    # fixed data, fixed OFT adapter, no SelectionAgent, no FeedbackAgent,
    # no EvaluationAgent, no AdapterAgent, and no Coordinator loop.
    if args.condition in ("static_sft", "odm"):
        import random, math

        # max_rounds may not yet be defined at this point because static_sft
        # bypasses the Coordinator path. Match the old round budget:
        # epochs * max_per_dataset / batch_size = 3 * 10000 / 8 = 3750.
        static_steps = max(1, (args.epochs * len(train_data)) // args.batch_size)
        if bench_rounds > 0:
            static_steps = min(static_steps, bench_rounds)

        if is_main:
            logger.info(
                "Running static_sft WITHOUT Coordinator | steps=%d | batch_size=%d",
                static_steps,
                args.batch_size,
            )

        rng = random.Random(args.seed)
        indices = list(range(len(train_data)))
        step = 0

        if args.condition == "odm":
            # ---- Online Data Mixing: EXP3 over data domains, reward = batch training loss ----
            domains = sorted(set(ex.get("source", "unknown") for ex in train_data))
            pools = {d: [i for i, ex in enumerate(train_data) if ex.get("source", "unknown") == d] for d in domains}
            cursors = {d: 0 for d in domains}
            for d in domains:
                rng.shuffle(pools[d])
            K = len(domains)
            prior = {d: len(pools[d]) / len(train_data) for d in domains}
            reward = {d: 0.0 for d in domains}
            alpha = 0.9                                   # moving-average smoothing of rewards
            import json as _json
            mix_log = open(os.path.join(args.output_dir, "odm_mixing.jsonl"), "w")
            counts = {d: 0 for d in domains}
            while step < static_steps:
                eps = min(1.0 / K, math.sqrt(math.log(K) / (K * (step + 1))))
                # start from the natural mixture (prior = domain size), as static_sft does, and
                # spread exploration in proportion to domain size so the 500 PubMedQA items are not oversampled
                z = [prior[d] * math.exp(eps * reward[d]) for d in domains]
                tot = sum(z)
                pi = [(1 - K * eps) * zi / tot + K * eps * prior[d] for zi, d in zip(z, domains)]
                d = rng.choices(domains, weights=pi)[0]
                k = domains.index(d)
                batch_idx = []
                for _ in range(args.batch_size):
                    if cursors[d] >= len(pools[d]):
                        rng.shuffle(pools[d]); cursors[d] = 0
                    batch_idx.append(pools[d][cursors[d]]); cursors[d] += 1
                batch = [train_data[i] for i in batch_idx]
                prompts = ["You are a clinical expert. Answer the following question.\n\n" + ex.get("input", "")
                           for ex in batch]
                loss = trainer.train_step_only(examples=batch, prompts=prompts, adapter_mode="oft")
                # importance-weighted reward update (EXP3), smoothed
                reward[d] = alpha * reward[d] + (1 - alpha) * (float(loss) / pi[k])
                counts[d] += 1
                step += 1
                if step % 50 == 0 and is_main:
                    logger.info("odm step %d / %d | pi=%s | counts=%s", step, static_steps,
                                {dd: round(pp, 3) for dd, pp in zip(domains, pi)}, counts)
                    mix_log.write(_json.dumps({"step": step, "pi": dict(zip(domains, pi)), "counts": counts}) + "\n")
                    mix_log.flush()
            mix_log.close()

        while args.condition == "static_sft" and step < static_steps:
            rng.shuffle(indices)

            for start in range(0, len(indices), args.batch_size):
                if step >= static_steps:
                    break

                batch_indices = indices[start:start + args.batch_size]
                batch = [train_data[i] for i in batch_indices]

                import json as _json
                _arch = ""
                try:
                    with open(args.model_path + "/config.json") as _f:
                        _arch = _json.load(_f).get("architectures", [""])[0]
                except: pass
                _is_gpt2 = "GPT2" in _arch
                prompts = [
                    (ex.get("input", "") + "\nAnswer:")
                    if _is_gpt2 else
                    ("You are a clinical expert. Answer the following question.\n\n"
                     + ex.get("input", ""))
                    for ex in batch
                ]
                _adapter = "lora" if _is_gpt2 else "oft"
                trainer.train_and_generate(
                    examples=batch,
                    prompts=prompts,
                    adapter_mode=_adapter,
                )

                step += 1

                if is_main and step % 50 == 0:
                    logger.info(
                        "static_sft direct training step %d / %d",
                        step,
                        static_steps,
                    )

        condition_dir = os.path.join(args.output_dir, args.condition)
        os.makedirs(condition_dir, exist_ok=True)

        adapter_path = os.path.join(condition_dir, "final_adapter")
        tokenizer_path = os.path.join(condition_dir, "tokenizer")

        if is_main:
            trainer._model.save_pretrained(adapter_path)
            trainer._tokenizer.save_pretrained(tokenizer_path)
            logger.info("static_sft final adapter saved to %s", adapter_path)
            prof = profiler.dump(os.path.join(condition_dir, "compute_profile.json"),
                                 condition=args.condition, rounds=static_steps,
                                 batch_size=args.batch_size)
            logger.info("Compute profile: %s", json.dumps(prof["seconds"]))

        if dist.is_available() and dist.is_initialized(): dist.barrier()

        trainer.cleanup()

        if is_main:
            logger.info("Condition static_sft complete without Coordinator.")

        return


    # --- Build agents ---
    from agents.task_classifier import TaskClassifier
    from agents.example_selection_agent import ExampleSelectionAgent, SelectionConfig
    from agents.evaluation_agent import EvaluationAgent, CriticConfig
    from agents.feedback_agent import FeedbackAgent, FeedbackConfig
    from agents.adapter_agent import AdapterAgent, StaticAdapterAgent
    from agents.coordinator import Coordinator, CoordinatorConfig

    CONDITIONS = {
        "static_sft":       {"selection_mode": "static",           "dynamic_adapter": False, "error_curriculum": False},
        "quality_only":     {"selection_mode": "quality",          "dynamic_adapter": False, "error_curriculum": False},
        "error_curriculum": {"selection_mode": "error_curriculum",  "dynamic_adapter": False, "error_curriculum": True},
        "dynamic_adapter":  {"selection_mode": "quality",          "dynamic_adapter": True,  "error_curriculum": False},
        "fmaca":            {"selection_mode": "error_curriculum",  "dynamic_adapter": True,  "error_curriculum": True},
        # Paper row "FMACA without contrastive loss": same loop as fmaca, contrastive off
        "static_coord":     {"selection_mode": "static",           "dynamic_adapter": False, "error_curriculum": False},
        "fmaca_nocontrastive": {"selection_mode": "error_curriculum", "dynamic_adapter": True, "error_curriculum": True},
    }
    cfg = CONDITIONS[args.condition]

    # Use total_steps instead of rounds to ensure fair comparison
    # Every condition trains for exactly the same number of gradient updates
    # regardless of how many examples the selection agent filters
    steps_per_epoch = max(1, len(train_data) // args.batch_size)
    max_rounds      = args.epochs * steps_per_epoch
    if bench_rounds > 0:
        max_rounds = min(max_rounds, bench_rounds)
    if is_main:
        logger.info(
            "Epochs=%d | steps_per_epoch=%d | total_rounds=%d",
            args.epochs, steps_per_epoch, max_rounds,
        )
        logger.info(
            "NOTE: All conditions run same number of rounds for fair comparison."
        )

    condition_dir = os.path.join(args.output_dir, args.condition)
    os.makedirs(condition_dir, exist_ok=True)

    coordinator = Coordinator(
        task_classifier  = TaskClassifier(),
        selection_agent  = ExampleSelectionAgent(config=SelectionConfig(
            min_input_length=10, min_output_length=1,
            quality_threshold=0.05, diversity_window=5000,
            easy_example_threshold=999,
            selection_mode=cfg["selection_mode"],
        ), seed=args.seed),
        evaluation_agent = EvaluationAgent(config=CriticConfig(
            correctness_weight=0.40, reasoning_weight=0.35, safety_weight=0.25,
        )),
        feedback_agent   = FeedbackAgent(config=FeedbackConfig(
            curriculum_lr=0.10 if cfg["error_curriculum"] else 0.0,
        ), seed=args.seed),
        adapter_agent    = AdapterAgent() if cfg["dynamic_adapter"] else StaticAdapterAgent(),
        model            = trainer,
        tokenizer        = None,
        config           = CoordinatorConfig(
            max_rounds=max_rounds,
            batch_size=args.batch_size,
            output_dir=condition_dir,
            # Disable early convergence for fair ablation runs.
            # Otherwise conditions stop after only a few rounds.
            convergence_threshold=-1.0,
            convergence_window=max_rounds + 1,
            save_every_n_rounds=100,
        ),
    )

    # --- Only rank 0 runs the coordinator loop ---
    # Other ranks participate via DeepSpeed collective ops inside trainer
    profiler.attach_coordinator(coordinator)
    if is_main:
        coordinator.run(train_data)

        # Save adapter
        try:
            adapter_path = os.path.join(condition_dir, "final_adapter")
            trainer._model.save_pretrained(adapter_path)
            trainer._tokenizer.save_pretrained(
                os.path.join(condition_dir, "tokenizer"))
            logger.info("Adapter saved to %s", adapter_path)
        except Exception as e:
            logger.warning("Could not save adapter: %s", e)
        prof = profiler.dump(os.path.join(condition_dir, "compute_profile.json"),
                             condition=args.condition, rounds=max_rounds,
                             batch_size=args.batch_size,
                             extra={"ce_negative_mode": os.environ.get("CE_NEGATIVE_MODE", "typed")})
        logger.info("Compute profile: %s", json.dumps(prof["seconds"]))

    # Sync all ranks before cleanup
    if dist.is_available() and dist.is_initialized(): dist.barrier()
    trainer.cleanup()
    if is_main:
        logger.info("Condition %s complete.", args.condition)

if __name__ == "__main__":
    main()
