"""
Example Selection Agent  (v6)
-----------------------------
Filters and scores training examples before they reach the optimizer.
Inspired by RAISE-style quality-based selection, extended with:
  - Per-example cognitive error history tracking
  - Curriculum-weight-aware selection scoring (Fix 3)
  - Error-stratified batch composition

Changes from v5:
  - All instruction-function / routing-hint logic removed
  - Imports shared taxonomy (COGNITIVE_ERROR_TYPES, CURRICULUM_CATEGORIES)
  - selection score now multiplied by curriculum_weight[example_curriculum_type]
  - example_curriculum_type inferred from task label
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter

from agents.taxonomy import (
    stable_example_id,
    COGNITIVE_ERROR_TYPES,
    CURRICULUM_CATEGORIES,
    TASK_TO_CURRICULUM,
    DEFAULT_CURRICULUM_WEIGHTS,
)

logger = logging.getLogger(__name__)


# Selection modes for clean ablation isolation
SELECTION_MODE_STATIC          = "static"           # no filtering: accept all valid length
SELECTION_MODE_QUALITY         = "quality"          # quality score only (no error/curriculum)
SELECTION_MODE_ERROR_CURRICULUM = "error_curriculum" # full FMACA: quality + error + curriculum


@dataclass
class SelectionConfig:
    min_input_length: int = 15           # words
    max_input_length: int = 512
    min_output_length: int = 5
    diversity_window: int = 100
    quality_threshold: float = 0.25
    max_selection_ratio: float = 0.90
    persistent_error_threshold: int = 2  # rounds → flag as hard
    easy_example_threshold: int = 3      # clean rounds → flag as easy
    curriculum_weight_strength: float = 0.4
    # Ablation mode — controls what selection score includes
    selection_mode: str = SELECTION_MODE_ERROR_CURRICULUM


@dataclass
class SelectionStats:
    total_seen: int = 0
    total_selected: int = 0
    rejected_too_short: int = 0
    rejected_too_long: int = 0
    rejected_duplicate: int = 0
    rejected_low_quality: int = 0
    flagged_persistent_errors: int = 0
    downweighted_easy_examples: int = 0
    per_task_selected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_task_rejected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_type_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))

    @property
    def selection_rate(self) -> float:
        return self.total_selected / max(self.total_seen, 1)


class ExampleSelectionAgent:
    """
    Quality + curriculum-aware example selection agent.

    Selection score = quality_score × curriculum_multiplier × error_history_weight

    curriculum_multiplier comes from FeedbackAgent's curriculum_weights,
    applied via update_curriculum_weights() each round.

    error_history_weight: hard examples get higher priority,
    easy examples get downweighted after repeated clean rounds.
    """

    def __init__(self, config: Optional[SelectionConfig] = None, seed: int = 42):
        self.config = config or SelectionConfig()
        self.rng    = np.random.default_rng(seed)
        self.stats  = SelectionStats()

        # Current curriculum weights (updated by FeedbackAgent via Coordinator)
        self._curriculum_weights: Dict[str, float] = DEFAULT_CURRICULUM_WEIGHTS.copy()

        # Duplicate detection
        self._recent_hashes: List[int] = []

        # Centroid tracking per task
        self._task_centroids: Dict[str, np.ndarray] = {}
        self._task_centroid_counts: Dict[str, int] = defaultdict(int)

        # Per-example error history
        self.example_error_log: Dict[str, List[str]] = {}

        # Hard / easy flags
        self._flagged_hard: set = set()
        self._flagged_easy: set = set()

        logger.info("ExampleSelectionAgent v6 initialized.")

    # ------------------------------------------------------------------
    # Public API — selection
    # ------------------------------------------------------------------

    def select(
        self,
        example: Dict,
        embedding: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float, str]:
        """
        Decide whether to include an example in the training batch.

        Returns:
            (accepted: bool, score: float, reason: str)
        """
        self.stats.total_seen += 1
        task = example.get("task", "medical_qa")

        # --- Hard filters ---
        input_len  = len(example.get("input", "").split())
        output_len = len(example.get("output", "").split())

        if input_len < self.config.min_input_length:
            self.stats.rejected_too_short += 1
            self.stats.per_task_rejected[task] += 1
            return False, 0.0, "too_short"

        if input_len > self.config.max_input_length:
            self.stats.rejected_too_long += 1
            self.stats.per_task_rejected[task] += 1
            return False, 0.0, "too_long"

        if output_len < self.config.min_output_length:
            self.stats.rejected_too_short += 1
            self.stats.per_task_rejected[task] += 1
            return False, 0.0, "output_too_short"

        # --- Duplicate detection ---
        input_hash = hash(example.get("input", ""))
        if False and input_hash in self._recent_hashes:  # disabled: allows re-sampling
            self.stats.rejected_duplicate += 1
            self.stats.per_task_rejected[task] += 1
            return False, 0.0, "duplicate"

        # --- Quality score ---
        length_score    = self._score_length(input_len)
        diversity_score = self._score_diversity(example)
        novelty_score   = self._score_novelty(task, embedding)
        quality_score   = (0.3 * length_score
                           + 0.4 * diversity_score
                           + 0.3 * novelty_score)

        # --- Composite selection score (mode-controlled for clean ablation) ---
        mode = self.config.selection_mode

        if mode == SELECTION_MODE_STATIC:
            # Static: accept all length-valid examples, no quality filter
            composite = 1.0

        elif mode == SELECTION_MODE_QUALITY:
            # Quality only: no error history, no curriculum weighting
            composite = quality_score

        else:
            # Full FMACA: quality + curriculum multiplier + error history weight
            curriculum_multiplier = self._curriculum_multiplier(example)
            error_weight          = self._error_history_weight(example)
            composite = (
                (1 - self.config.curriculum_weight_strength) * quality_score
                + self.config.curriculum_weight_strength * curriculum_multiplier
            ) * error_weight

        if composite < self.config.quality_threshold:
            self.stats.rejected_low_quality += 1
            self.stats.per_task_rejected[task] += 1
            return False, composite, "low_quality"

        self._update_state(input_hash, task, embedding)
        self.stats.total_selected += 1
        self.stats.per_task_selected[task] += 1
        return True, composite, "accepted"

    def filter_batch(
        self,
        examples: List[Dict],
        embeddings: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[Dict], List[float]]:
        accepted, scores = [], []
        for i, ex in enumerate(examples):
            emb = embeddings[i] if embeddings else None
            ok, score, _ = self.select(ex, emb)
            if ok:
                accepted.append(ex)
                scores.append(score)
        return accepted, scores

    def update_curriculum_weights(self, weights: Dict[str, float]) -> None:
        """
        Receive updated curriculum weights from FeedbackAgent via Coordinator.
        These directly influence example selection scores next round.
        """
        self._curriculum_weights = weights.copy()
        logger.info(
            "Curriculum weights updated: %s",
            {k: f"{v:.3f}" for k, v in weights.items()},
        )

    def update_criteria(self, criteria: Dict[str, float]) -> None:
        """Legacy compatibility — maps quality criteria adjustments."""
        pass  # criteria are now fixed; curriculum weights do the adaptation

    # ------------------------------------------------------------------
    # Cognitive error tracking API
    # ------------------------------------------------------------------

    def log_error(self, example_id: str, error_type: str) -> None:
        """
        Record cognitive error type for this example.
        Uses shared taxonomy — unrecognised types default to 'none'.
        """
        if error_type not in COGNITIVE_ERROR_TYPES:
            logger.debug(
                "Unknown error type '%s' for example %s → mapped to 'none'",
                error_type, example_id[:8],
            )
            error_type = "none"

        if example_id not in self.example_error_log:
            self.example_error_log[example_id] = []

        self.example_error_log[example_id].append(error_type)
        self.stats.error_type_counts[error_type] += 1
        self._update_flags(example_id)

    def get_error_profile(self, example_id: str) -> Dict:
        errors   = self.example_error_log.get(example_id, [])
        non_none = [e for e in errors if e != "none"]
        if not errors:
            return {
                "total_exposures": 0, "error_counts": {},
                "dominant_error": "none", "error_rate": 0.0,
                "is_hard": False, "is_easy": False,
            }
        return {
            "total_exposures": len(errors),
            "error_counts":    dict(Counter(errors)),
            "dominant_error":  Counter(errors).most_common(1)[0][0],
            "error_rate":      len(non_none) / len(errors),
            "is_hard":         example_id in self._flagged_hard,
            "is_easy":         example_id in self._flagged_easy,
        }

    def get_aggregate_error_report(self) -> Dict:
        all_errors  = [e for errs in self.example_error_log.values() for e in errs]
        if not all_errors:
            return {"message": "No error history yet."}
        error_counts = Counter(all_errors)
        total        = len(all_errors)
        return {
            "total_observations":     total,
            "unique_examples":        len(self.example_error_log),
            "per_type": {
                et: {"count": error_counts.get(et, 0),
                     "fraction": error_counts.get(et, 0) / total}
                for et in COGNITIVE_ERROR_TYPES
            },
            "hard_examples":          len(self._flagged_hard),
            "easy_examples":          len(self._flagged_easy),
            "most_common_error":      error_counts.most_common(1)[0],
        }

    def get_stats(self) -> Dict:
        return {
            "total_seen":             self.stats.total_seen,
            "total_selected":         self.stats.total_selected,
            "selection_rate":         self.stats.selection_rate,
            "rejected_too_short":     self.stats.rejected_too_short,
            "rejected_too_long":      self.stats.rejected_too_long,
            "rejected_duplicate":     self.stats.rejected_duplicate,
            "rejected_low_quality":   self.stats.rejected_low_quality,
            "flagged_hard":           len(self._flagged_hard),
            "flagged_easy":           len(self._flagged_easy),
            "per_task_selected":      dict(self.stats.per_task_selected),
            "error_type_counts":      dict(self.stats.error_type_counts),
            "curriculum_weights":     self._curriculum_weights.copy(),
        }

    @staticmethod
    def _example_id(example: Dict) -> str:
        """Stable, reproducible ID — uses SHA-256, not Python hash()."""
        return stable_example_id(example.get("input", ""))

    # ------------------------------------------------------------------
    # Private — scoring
    # ------------------------------------------------------------------

    def _score_length(self, input_len: int) -> float:
        ideal_min, ideal_max = 30, 300
        if ideal_min <= input_len <= ideal_max:
            return 1.0
        if input_len < ideal_min:
            return input_len / ideal_min
        return max(0.0, 1.0 - (input_len - ideal_max) / ideal_max)

    def _score_diversity(self, example: Dict) -> float:
        tokens = example.get("input", "").lower().split()
        if not tokens:
            return 0.0
        ttr = len(set(tokens)) / len(tokens)
        return min(1.0, ttr / 0.5)

    def _score_novelty(self, task: str, embedding: Optional[np.ndarray]) -> float:
        if embedding is None or task not in self._task_centroids:
            return 0.5
        dist = float(np.linalg.norm(embedding - self._task_centroids[task]))
        return min(1.0, dist / 5.0)

    def _curriculum_multiplier(self, example: Dict) -> float:
        """
        Return curriculum weight for this example's category.
        Examples whose category is currently prioritised score higher.
        Inferred from task label using TASK_TO_CURRICULUM mapping.
        """
        task     = example.get("task", "medical_qa")
        category = TASK_TO_CURRICULUM.get(task, "routine")
        return self._curriculum_weights.get(category, 1.0 / len(CURRICULUM_CATEGORIES))

    def _error_history_weight(self, example: Dict) -> float:
        """
        Weight based on example's error history:
          hard examples (persistent errors) : upweighted (1.3)
          easy examples (no errors)         : downweighted (0.6)
          unknown / mixed                   : neutral (1.0)
        """
        eid = self._example_id(example)
        if eid in self._flagged_hard:
            return 1.3  # model struggles → prioritise
        if eid in self._flagged_easy:
            return 0.6  # model has mastered → deprioritise
        return 1.0

    # ------------------------------------------------------------------
    # Private — state updates
    # ------------------------------------------------------------------

    def _update_state(
        self,
        input_hash: int,
        task: str,
        embedding: Optional[np.ndarray],
    ) -> None:
        self._recent_hashes.append(input_hash)
        if len(self._recent_hashes) > self.config.diversity_window:
            self._recent_hashes.pop(0)

        if embedding is not None:
            count = self._task_centroid_counts[task]
            if task not in self._task_centroids:
                self._task_centroids[task] = embedding.copy()
            else:
                self._task_centroids[task] = (
                    self._task_centroids[task] * count + embedding
                ) / (count + 1)
            self._task_centroid_counts[task] += 1

    def _update_flags(self, example_id: str) -> None:
        errors   = self.example_error_log.get(example_id, [])
        non_none = [e for e in errors if e != "none"]

        if len(non_none) >= self.config.persistent_error_threshold:
            if example_id not in self._flagged_hard:
                self._flagged_hard.add(example_id)
                self.stats.flagged_persistent_errors += 1
                dominant = Counter(non_none).most_common(1)[0][0]
                logger.info(
                    "Example %s flagged as HARD | dominant_error=%s",
                    example_id[:8], dominant,
                )

        if (len(errors) >= self.config.easy_example_threshold
                and len(non_none) == 0):
            if example_id not in self._flagged_easy:
                self._flagged_easy.add(example_id)
                self.stats.downweighted_easy_examples += 1
