"""
Coordinator / Orchestrator  (v5)
--------------------------------
Training-time adaptation loop for FMACA.

Architecture (v5 — no instruction functions, no RouterAgent):

  Round N:
    1. TaskClassifier   identifies task type per example
    2. SelectionAgent   filters by quality + error history + curriculum weights
    3. LLM fine-tuning  using adapter config selected in round N-1
    4. CriticAgent      scores output, classifies cognitive error type
    5. SelectionAgent   logs per-example error (builds error history)
    6. FeedbackAgent    updates curriculum weights + recommends adapter mode
    7. AdapterAgent     selects OFT/BOFT config for round N+1

Key fix from v4: adapter is selected AT END of round N,
applied AT START of round N+1. Not selected after training.

The Coordinator is the "Residency Program Director":
  - Does not generate clinical answers
  - Coordinates specialists
  - Updates the curriculum based on resident's error patterns
"""

import os
import json
import random
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter

from agents.task_classifier import TaskClassifier
from agents.example_selection_agent import ExampleSelectionAgent
from agents.evaluation_agent import EvaluationAgent, COGNITIVE_ERROR_TYPES
from agents.feedback_agent import FeedbackAgent, FeedbackSignal
from agents.adapter_agent import AdapterAgent, StaticAdapterAgent

logger = logging.getLogger(__name__)

CE_NEGATIVE_MODES = ("typed", "random", "shuffled_types")
NON_NONE_ERRORS = [e for e in COGNITIVE_ERROR_TYPES if e != "none"]


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (data sampling and contrastive pair sampling)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class CoordinatorConfig:
    max_rounds: int = 10
    batch_size: int = 32
    convergence_threshold: float = 0.01
    convergence_window: int = 3
    save_every_n_rounds: int = 2
    output_dir: str = "./fmaca_output"


@dataclass
class RoundResult:
    round_num: int
    task: str
    n_examples_seen: int
    n_examples_selected: int
    selection_rate: float
    avg_critic_scores: Dict[str, float]
    error_counts: Dict[str, int]
    dominant_error: str
    feedback_signal: str
    curriculum_weights: Dict[str, float]
    adapter_mode_next: str   # adapter for NEXT round (fixed sequencing)
    converged: bool = False


class Coordinator:
    """
    FMACA v5 training-time orchestrator.
    No RouterAgent. No instruction functions.
    """

    def __init__(
        self,
        task_classifier: TaskClassifier,
        selection_agent: ExampleSelectionAgent,
        evaluation_agent: EvaluationAgent,
        feedback_agent: FeedbackAgent,
        adapter_agent=None,
        model=None,
        tokenizer=None,
        config: Optional[CoordinatorConfig] = None,
    ):
        self.task_classifier  = task_classifier
        self.selection_agent  = selection_agent
        self.evaluation_agent = evaluation_agent
        self.feedback_agent   = feedback_agent
        self.adapter_agent    = adapter_agent or StaticAdapterAgent()
        self.model            = model
        self.tokenizer        = tokenizer
        self.config           = config or CoordinatorConfig()

        # Per-task adapter state (Fix 4: per-task, not shared across tasks)
        # active: applied at START of each round (selected previous round)
        # next:   selected at END of each round, applied next round
        self._active_adapter_by_task: Dict[str, str] = {}
        self._next_adapter_by_task:   Dict[str, str] = {}
        # Per-task error history for task-specific adapter voting
        self._task_error_history: Dict[str, List[str]] = {}

        self._round_results: List[RoundResult] = []
        self._converged = False

        # ---- Revision ablation: contrastive negative mode ----
        # CE_NEGATIVE_MODE=typed           paper behaviour (critic-flagged own outputs)
        # CE_NEGATIVE_MODE=random          same pair count, negative = another question's output
        # CE_NEGATIVE_MODE=shuffled_types  critic error labels replaced by random taxonomy labels
        self.ce_negative_mode = os.environ.get("CE_NEGATIVE_MODE", "typed").strip().lower()
        if self.ce_negative_mode not in CE_NEGATIVE_MODES:
            raise ValueError(
                f"CE_NEGATIVE_MODE={self.ce_negative_mode!r}; expected one of {CE_NEGATIVE_MODES}"
            )
        seed_env = os.environ.get("FMACA_SEED")
        self.seed = int(seed_env) if seed_env is not None else 42
        if seed_env is not None:
            seed_everything(self.seed)
        # Separate RNG for negative sampling so it never shifts the data-sampling stream
        self._ce_rng = random.Random(self.seed + 1009)
        self._neg_pool: List[Dict[str, str]] = []   # recent generated outputs (other questions)
        self._neg_pool_max = 256
        self._ce_log_path = os.path.join(self.config.output_dir, "ce_negatives.jsonl")
        self._ce_totals: Counter = Counter()
        logger.info(
            "Contrastive negative mode=%s | seed=%s (%s)",
            self.ce_negative_mode, self.seed,
            "seeded" if seed_env is not None else "UNSEEDED, set FMACA_SEED",
        )

        os.makedirs(self.config.output_dir, exist_ok=True)
        logger.info("Coordinator v5 initialized. Output: %s", self.config.output_dir)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        train_data: List[Dict],
        embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[RoundResult]:

        logger.info(
            "FMACA v5 loop | rounds=%d | data=%d examples",
            self.config.max_rounds, len(train_data),
        )

        # Classify tasks upfront if not already labeled
        for ex in train_data:
            if "task" not in ex:
                ex["task"] = self.task_classifier.classify(ex)

        for round_num in range(1, self.config.max_rounds + 1):
            logger.info("=== Round %d / %d ===", round_num, self.config.max_rounds)

            # Promote next-round adapters to active at start of each round
            for t, mode in self._next_adapter_by_task.items():
                self._active_adapter_by_task[t] = mode

            # Sample batch
            batch = self._sample_batch(train_data)
            batch_embeddings = None
            if embeddings:
                # align embeddings with batch (by index)
                indices = [train_data.index(ex) for ex in batch]
                batch_embeddings = [embeddings[i] for i in indices]

            # Group by task
            # sorted(): set order depends on PYTHONHASHSEED and made runs irreproducible
            tasks_in_batch = sorted(set(ex.get("task", "medical_qa") for ex in batch))

            for task in tasks_in_batch:
                task_batch = [ex for ex in batch if ex.get("task") == task]
                task_embs  = None  # embeddings per task not tracked in v5

                result = self._run_round(round_num, task, task_batch, task_embs)
                self._round_results.append(result)

                if result.converged:
                    self._converged = True

            if round_num % self.config.save_every_n_rounds == 0:
                self._save_results(round_num)

            if self._converged:
                break

        self._save_results("final")
        logger.info("FMACA loop complete. %d round-results logged.", len(self._round_results))
        return self._round_results

    # ------------------------------------------------------------------
    # Single round
    # ------------------------------------------------------------------

    def _run_round(
        self,
        round_num: int,
        task: str,
        examples: List[Dict],
        embeddings: Optional[List[np.ndarray]],
    ) -> RoundResult:

        # ---- Step 1: Apply adapter from PREVIOUS round (per-task) ----
        # Active adapter for this task was selected at end of last round.
        # Default to OFT for first round.
        current_adapter = self._active_adapter_by_task.get(task, "oft")
        logger.info(
            "Round %d | %s | using adapter=%s (selected last round)",
            round_num, task, current_adapter.upper(),
        )

        # ---- Step 2: Example Selection ----
        # Use curriculum weights to stratify batch
        curriculum_weights = self.feedback_agent.get_curriculum_weights()
        selected = []
        for i, ex in enumerate(examples):
            emb = embeddings[i] if embeddings else None
            accepted, score, reason = self.selection_agent.select(ex, emb)
            if accepted:
                selected.append(ex)

        if not selected:
            logger.warning(
                "Round %d task %s: no examples passed selection — using unfiltered task batch fallback.",
                round_num, task
            )
            # Fallback: keep the update budget fair by training on the original task batch.
            # This prevents coordinator-based methods from wasting rounds when the
            # SelectionAgent rejects all examples in a small task batch.
            selected = list(examples)

        selection_rate = len(selected) / max(len(examples), 1)
        logger.info(
            "Round %d | %s | selected %d/%d (%.0f%%)",
            round_num, task, len(selected), len(examples), 100 * selection_rate,
        )

        # ---- Step 3: LLM fine-tuning step ----
        outputs = self._finetune_step(selected, round_num=round_num)

        # ---- Step 4: Critic evaluation ----
        all_scores, avg_scores = self.evaluation_agent.evaluate_batch(
            outputs=outputs,
            golds=[ex.get("output", "") for ex in selected],
            tasks=[task] * len(selected),
            input_texts=[ex.get("input", "") for ex in selected],
        )

        # ---- Step 4b (shuffled_types only): replace error TYPE, keep error/no-error ----
        # Keeps the contrastive gate and hard/easy flags identical to typed mode,
        # so only the information carried by the six-way type is destroyed.
        original_types = [s.failure_type for s in all_scores]
        if self.ce_negative_mode == "shuffled_types":
            for s in all_scores:
                if s.failure_type != "none":
                    s.failure_type = self._ce_rng.choice(NON_NONE_ERRORS)

        # ---- Step 5: Log cognitive errors + contrastive pairs ----
        can_add = (self.model is not None and hasattr(self.model, "add_contrastive_pair"))
        round_generated = []          # generated outputs this sub-batch (for the negative pool)
        n_generated = n_flagged = n_added = n_skipped = 0
        used_types: Counter = Counter()

        for ex, score, output in zip(selected, all_scores, outputs):
            eid = self.selection_agent._example_id(ex)
            self.selection_agent.log_error(eid, score.failure_type)

            gold = ex.get("output", "")
            is_generated = output.strip() != gold.strip()
            if is_generated:
                n_generated += 1
                round_generated.append({"eid": eid, "text": output})

            # Gate is identical in all modes: critic error detected on a real generated output
            if not (can_add and is_generated and score.failure_type != "none"):
                continue
            n_flagged += 1
            prompt = self._contrastive_prompt(ex)

            if self.ce_negative_mode == "random":
                wrong = self._sample_random_negative(eid, gold, round_generated, selected, outputs)
                if wrong is None:
                    n_skipped += 1
                    continue
                error_type = "random_negative"
            else:
                wrong = output
                error_type = score.failure_type

            # wrong  = typed: real model output flagged by critic; random: another question's output
            # chosen = gold answer (always safe and correct)
            self.model.add_contrastive_pair(
                prompt=prompt,
                wrong_output=wrong,
                chosen_output=gold,
                error_type=error_type,
                example_id=eid,
            )
            n_added += 1
            used_types[error_type] += 1

        # Update rolling negative pool after this sub-batch
        if round_generated:
            self._neg_pool.extend(round_generated)
            self._neg_pool = self._neg_pool[-self._neg_pool_max:]

        if n_generated:
            self._log_ce_round(round_num, task, len(selected), n_generated,
                               n_flagged, n_added, n_skipped,
                               Counter(original_types), used_types)

        # ---- Step 6: Feedback — update curriculum weights ----
        error_counts = Counter(s.failure_type for s in all_scores)
        signal, updates = self.feedback_agent.process(
            critic_scores=avg_scores,
            error_counts=dict(error_counts),
            round_num=round_num,
        )

        dominant_error  = updates["dominant_error"]
        new_curriculum  = updates["curriculum_weights"]
        adapter_rec     = updates["adapter_recommendation"]

        # Always update curriculum weights in SelectionAgent (Fix 3)
        # This is what makes example selection curriculum-aware each round
        self.selection_agent.update_curriculum_weights(new_curriculum)

        # Additional criteria update on stagnation
        if signal == FeedbackSignal.BOTH:
            self.selection_agent.update_criteria(
                self._curriculum_to_selection_criteria(new_curriculum)
            )

        # ---- Step 7: AdapterAgent selects config for NEXT round ----
        # Use task-specific error history (not global) for adapter voting
        if task not in self._task_error_history:
            self._task_error_history[task] = []
        self._task_error_history[task].append(dominant_error)

        next_mode, next_config = self.adapter_agent.select(
            dominant_error=dominant_error,
            task=task,
            round_num=round_num,
            error_history=self._task_error_history[task],  # task-specific only
        )
        # Store per-task: will become active at start of next round
        self._next_adapter_by_task[task] = next_mode

        converged = self._check_convergence()

        return RoundResult(
            round_num=round_num,
            task=task,
            n_examples_seen=len(examples),
            n_examples_selected=len(selected),
            selection_rate=selection_rate,
            avg_critic_scores=avg_scores,
            error_counts=dict(error_counts),
            dominant_error=dominant_error,
            feedback_signal=signal.value,
            curriculum_weights=new_curriculum,
            adapter_mode_next=next_mode,   # will be active next round
            converged=converged,
        )

    # ------------------------------------------------------------------
    # LLM stub
    # ------------------------------------------------------------------

    def _finetune_step(self, examples: List[Dict], round_num: int = 0) -> List[str]:
        GENERATE_EVERY = 500  # generate real outputs ~once per epoch
        """
        Fine-tuning step — separated into train + generate.

        1. train_step_only(): SFT on (prompt, gold) + contrastive loss
        2. generate_outputs(): generate REAL model outputs for critic

        Returns real model outputs so critic detects actual errors.
        Falls back to gold outputs only if trainer unavailable.
        """
        if self.model is None:
            return [ex.get("output", "") for ex in examples]

        try:
            from trainer import FMACATrainer
            if isinstance(self.model, FMACATrainer):
                prompts = [
                    (ex.get("input","") + "\nAnswer:")
                    if (hasattr(self.model, "_model") and
                        "GPT2" in getattr(getattr(self.model, "_model", None),
                        "config", type("c", (), {"architectures":[""]})()).architectures[0])
                    else
                    (ex.get("input","") + "\nAnswer:")
                    if getattr(self.model, "is_gpt2", False) else
                    "You are a clinical expert. Answer the following question.\n\n"
                    + ex.get("input", "")
                    for ex in examples
                ]
                adapter_mode = self._active_adapter_by_task.get(
                    examples[0].get("task", "medical_qa"), "oft"
                )

                # Use new separated API if available
                if hasattr(self.model, "train_step_only"):
                    # Step 1: Train (SFT + contrastive)
                    self.model.train_step_only(
                        examples=examples,
                        prompts=prompts,
                        adapter_mode=adapter_mode,
                    )
                    # Step 2: Generate REAL outputs only every GENERATE_EVERY rounds
                    if round_num % GENERATE_EVERY == 1:
                        return self.model.generate_outputs(prompts)
                    else:
                        return [ex.get("output", "") for ex in examples]
                else:
                    # Fallback: old API (returns gold)
                    return self.model.train_and_generate(
                        examples=examples,
                        prompts=prompts,
                        adapter_mode=adapter_mode,
                    )

        except Exception as e:
            logger.warning("Trainer step failed: %s — returning gold.", e)

        return [ex.get("output", "") for ex in examples]

    # ------------------------------------------------------------------
    # Contrastive negative helpers (revision ablation)
    # ------------------------------------------------------------------

    def _contrastive_prompt(self, ex: Dict) -> str:
        """Same prompt expression as the original Step 5 (kept verbatim)."""
        return (
            (ex.get("input","") + "\nAnswer:")
            if (hasattr(self.model, "_model") and
                "GPT2" in getattr(getattr(self.model, "_model", None),
                "config", type("c", (), {"architectures":[""]})()).architectures[0])
            else
            (ex.get("input","") + "\nAnswer:")
            if getattr(self.model, "is_gpt2", False) else
            "You are a clinical expert. Answer the following question.\n\n"
            + ex.get("input", "")
        )

    def _sample_random_negative(
        self,
        eid: str,
        gold: str,
        round_generated: List[Dict[str, str]],
        selected: List[Dict],
        outputs: List[str],
    ) -> Optional[str]:
        """
        Random negative: a real model output generated for a DIFFERENT question.
        Candidates: all generated outputs in this sub-batch plus the rolling pool
        of earlier generation rounds. Excludes the same question and any text
        identical to this question's gold answer.
        """
        current = []
        for ex_j, out_j in zip(selected, outputs):
            if out_j.strip() != ex_j.get("output", "").strip():
                current.append({"eid": self.selection_agent._example_id(ex_j), "text": out_j})
        candidates = [
            c for c in current + self._neg_pool
            if c["eid"] != eid and c["text"].strip() and c["text"].strip() != gold.strip()
        ]
        if not candidates:
            return None
        return self._ce_rng.choice(candidates)["text"]

    def _log_ce_round(self, round_num, task, n_selected, n_generated,
                      n_flagged, n_added, n_skipped, critic_types, used_types) -> None:
        self._ce_totals.update({
            "generated": n_generated, "flagged": n_flagged,
            "added": n_added, "skipped": n_skipped,
        })
        rec = {
            "round": round_num, "task": task, "mode": self.ce_negative_mode,
            "seed": self.seed, "n_selected": n_selected,
            "n_generated": n_generated, "n_flagged": n_flagged,
            "n_added": n_added, "n_skipped": n_skipped,
            "critic_types": dict(critic_types), "pair_types": dict(used_types),
            "totals": dict(self._ce_totals),
        }
        try:
            with open(self._ce_log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            logger.warning("Could not write %s: %s", self._ce_log_path, e)
        logger.info(
            "CE negatives | mode=%s | round=%d | %s | generated=%d flagged=%d added=%d skipped=%d",
            self.ce_negative_mode, round_num, task, n_generated, n_flagged, n_added, n_skipped,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_batch(self, data: List[Dict]) -> List[Dict]:
        n = min(self.config.batch_size, len(data))
        idx = np.random.choice(len(data), size=n, replace=False)
        return [data[i] for i in idx]

    def _curriculum_to_selection_criteria(
        self, curriculum_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Map curriculum category weights to SelectionAgent criteria weights.
        Curriculum categories with high weight -> preference for diverse, novel examples.
        """
        # Higher curriculum complexity → more novelty and diversity needed
        complexity = (
            curriculum_weights.get("differential_reasoning", 0.2)
            + curriculum_weights.get("multi_step_reasoning", 0.2)
            + curriculum_weights.get("misleading_cue", 0.15)
        )
        return {
            "length":    0.25,
            "diversity": min(0.5, 0.25 + complexity * 0.5),
            "novelty":   max(0.1, 0.5 - complexity * 0.3),
        }

    def _check_convergence(self) -> bool:
        history = self.feedback_agent.get_history()
        scores  = history.get("round_scores", [])
        w = self.config.convergence_window
        if len(scores) < w:
            return False
        recent = scores[-w:]
        return (max(recent) - min(recent)) < self.config.convergence_threshold

    def _empty_result(self, round_num, task, n_seen) -> RoundResult:
        return RoundResult(
            round_num=round_num,
            task=task,
            n_examples_seen=n_seen,
            n_examples_selected=0,
            selection_rate=0.0,
            avg_critic_scores={},
            error_counts={},
            dominant_error="none",
            feedback_signal="none",
            curriculum_weights=self.feedback_agent.get_curriculum_weights(),
            adapter_mode_next=self._active_adapter_by_task.get(task, 'oft'),
        )

    def _save_results(self, label) -> None:
        path = os.path.join(self.config.output_dir, f"results_round_{label}.json")
        data = []
        for r in self._round_results:
            data.append({
                "round":            r.round_num,
                "task":             r.task,
                "n_seen":           r.n_examples_seen,
                "n_selected":       r.n_examples_selected,
                "selection_rate":   r.selection_rate,
                "critic_scores":    r.avg_critic_scores,
                "error_counts":     r.error_counts,
                "dominant_error":   r.dominant_error,
                "feedback_signal":  r.feedback_signal,
                "curriculum_weights": r.curriculum_weights,
                "adapter_mode_next": r.adapter_mode_next,
                "converged":        r.converged,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %s", path)

    def get_summary(self) -> Dict:
        return {
            "total_rounds":          len(self._round_results),
            "converged":             self._converged,
            "selection_stats":       self.selection_agent.get_stats(),
            "task_classifier_stats": self.task_classifier.get_stats(),
            "feedback_history":      self.feedback_agent.get_history(),
            "adapter_stats":         self.adapter_agent.get_stats(),
            "cognitive_error_report": self.selection_agent.get_aggregate_error_report(),
            "final_curriculum_weights": self.feedback_agent.get_curriculum_weights(),
        }
