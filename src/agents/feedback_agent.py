"""
Feedback Agent  (v5 — curriculum weights, no routing probabilities)
-------------------------------------------------------------------
Converts critic cognitive error signals into curriculum weight updates
for the ExampleSelectionAgent and adapter mode signals for AdapterAgent.

This is the core of FMACA's closed adaptation loop:

  Critic error signal → FeedbackAgent → curriculum reweighting
                                      → adapter mode recommendation

Curriculum weight vector controls what *type* of examples the
SelectionAgent prioritises in the next round:

  differential_reasoning : cases requiring broad differential diagnosis
  misleading_cue         : cases with contradictory / late-breaking evidence
  factual_evidence       : factual/evidence-heavy questions
  multi_step_reasoning   : multi-step clinical reasoning chains
  safety_critical        : cases with red flags / safety-critical findings
  routine                : standard cases (downweighted when model improves)

Error-to-curriculum mapping (clinically grounded):
  anchoring          → ↑ misleading_cue (model needs contradictory evidence cases)
  premature_closure  → ↑ differential_reasoning (model needs breadth practice)
  knowledge_gap      → ↑ factual_evidence + BOFT adapter
  reasoning_failure  → ↑ multi_step_reasoning + BOFT adapter
  safety_omission    → ↑ safety_critical
  none (repeatedly)  → ↓ routine (downweight easy cases)

Changes from v4:
  - Removed routing probability update (no instruction functions)
  - Added curriculum_weights as the primary feedback target
  - FeedbackSignal now distinguishes CURRICULUM vs ADAPTER vs BOTH
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import Counter

from agents.taxonomy import (
    COGNITIVE_ERROR_TYPES,
    CURRICULUM_CATEGORIES,
    ERROR_TO_CURRICULUM,
    BOFT_ERRORS,
    DEFAULT_CURRICULUM_WEIGHTS,
)

logger = logging.getLogger(__name__)


class FeedbackSignal(Enum):
    CURRICULUM_UPDATE = "curriculum_update"
    ADAPTER_UPDATE    = "adapter_update"
    BOTH              = "both"
    NONE              = "none"


@dataclass
class FeedbackConfig:
    # Curriculum update step size
    curriculum_lr: float = 0.10

    # Minimum weight per curriculum category (prevents starvation)
    min_curriculum_weight: float = 0.05

    # Stagnation detection
    stagnation_window: int = 3
    stagnation_tolerance: float = 0.02

    # Entropy regularization — keep curriculum diverse
    entropy_lambda: float = 0.05

    # Score threshold below which adapter upgrade is triggered
    adapter_upgrade_threshold: float = 0.55



class FeedbackAgent:
    """
    Curriculum-driven feedback agent for FMACA.

    Receives cognitive error distribution from critic each round,
    updates curriculum type weights, and recommends adapter mode.
    """

    def __init__(
        self,
        config: Optional[FeedbackConfig] = None,
        seed: int = 42,
    ):
        self.config = config or FeedbackConfig()
        self.rng    = np.random.default_rng(seed)

        # Current curriculum weights (updated each round)
        self.curriculum_weights = DEFAULT_CURRICULUM_WEIGHTS.copy()

        # History
        self._round_scores:       List[float] = []
        self._error_history:      List[str]   = []
        self._curriculum_history: List[Dict]  = []
        self._adapter_history:    List[str]   = []

        logger.info("FeedbackAgent initialized (v5: curriculum weights).")

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def process(
        self,
        critic_scores: Dict[str, float],
        error_counts: Dict[str, int],
        round_num: int,
    ) -> Tuple[FeedbackSignal, Dict]:
        """
        Process critic output and produce curriculum + adapter updates.

        Args:
            critic_scores : avg scores {"correctness", "reasoning", "safety", "composite"}
            error_counts  : Counter of cognitive error types this round
            round_num     : current training round

        Returns:
            (signal, updates)
            updates keys: "curriculum_weights", "adapter_recommendation"
        """
        composite = critic_scores.get("composite", 0.5)
        self._round_scores.append(composite)

        # Dominant error type this round
        # Prefer non-none errors: if meaningful error rate exists, use it
        # so that minority clinical errors are not drowned out by "none"
        non_none_counts = {k: v for k, v in error_counts.items() if k != "none"}
        total_errors    = sum(error_counts.values())
        non_none_total  = sum(non_none_counts.values())
        non_none_rate   = non_none_total / max(total_errors, 1)

        if non_none_rate >= 0.25 and non_none_counts:
            dominant_error = max(non_none_counts, key=non_none_counts.get)
        elif error_counts:
            dominant_error = max(error_counts, key=error_counts.get)
        else:
            dominant_error = "none"
        self._error_history.append(dominant_error)

        # Update curriculum weights
        new_weights = self._update_curriculum(dominant_error, error_counts)
        self._curriculum_history.append(new_weights.copy())

        # Adapter recommendation
        adapter_rec = self._recommend_adapter(
            dominant_error, composite, error_counts
        )
        self._adapter_history.append(adapter_rec)

        updates = {
            "curriculum_weights":    new_weights,
            "adapter_recommendation": adapter_rec,
            "dominant_error":        dominant_error,
            "composite_score":       composite,
        }

        # Determine signal type
        stagnating = self._is_stagnating()
        if stagnating or composite < self.config.adapter_upgrade_threshold:
            signal = FeedbackSignal.BOTH
        else:
            signal = FeedbackSignal.CURRICULUM_UPDATE

        logger.info(
            "Round %d | composite=%.3f | dominant_error=%s | "
            "adapter_rec=%s | signal=%s",
            round_num, composite, dominant_error, adapter_rec, signal.value,
        )
        logger.info(
            "Curriculum weights: %s",
            {k: f"{v:.3f}" for k, v in new_weights.items()},
        )

        return signal, updates

    def get_curriculum_weights(self) -> Dict[str, float]:
        """Return current curriculum weight distribution."""
        return self.curriculum_weights.copy()

    def get_history(self) -> Dict:
        return {
            "round_scores":           self._round_scores,
            "error_history":          self._error_history,
            "adapter_history":        self._adapter_history,
            "mean_composite":         float(np.mean(self._round_scores))
                                      if self._round_scores else 0.0,
            "dominant_error_overall": Counter(self._error_history).most_common(1)[0][0]
                                      if self._error_history else "none",
        }

    # ------------------------------------------------------------------
    # Curriculum update
    # ------------------------------------------------------------------

    def _update_curriculum(
        self,
        dominant_error: str,
        error_counts: Dict[str, int],
    ) -> Dict[str, float]:
        """
        Update curriculum weights based on error distribution.

        Upweights the category that targets the dominant error.
        Downweights routine if model is performing well (few errors).
        Applies entropy regularization to keep curriculum diverse.
        """
        weights = self.curriculum_weights.copy()
        lr      = self.config.curriculum_lr

        # Target category for dominant error
        target_category = ERROR_TO_CURRICULUM.get(dominant_error, "routine")

        # Upweight target, downweight proportionally from others
        if target_category != "routine":
            weights[target_category] = min(0.5, weights[target_category] + lr)
            # Downweight routine (least useful when errors persist)
            weights["routine"] = max(
                self.config.min_curriculum_weight,
                weights["routine"] - lr * 0.5,
            )
        else:
            # Dominant error is "none" — downweight routine (model has mastered it)
            weights["routine"] = max(
                self.config.min_curriculum_weight,
                weights["routine"] - lr * 0.3,
            )
            # Spread to harder categories
            for cat in ["differential_reasoning", "multi_step_reasoning"]:
                weights[cat] = min(0.4, weights[cat] + lr * 0.15)

        # Secondary upweights from error distribution
        total_errors = sum(error_counts.values())
        if total_errors > 0:
            for error, count in error_counts.items():
                if error == "none":
                    continue
                cat   = ERROR_TO_CURRICULUM.get(error, "routine")
                boost = lr * 0.3 * (count / total_errors)
                weights[cat] = min(0.5, weights[cat] + boost)

        # Entropy regularization — pull toward uniform
        n       = len(weights)
        uniform = 1.0 / n
        weights = {
            k: (1 - self.config.entropy_lambda) * v
               + self.config.entropy_lambda * uniform
            for k, v in weights.items()
        }

        # Enforce minimum and renormalize
        weights = {k: max(self.config.min_curriculum_weight, v)
                   for k, v in weights.items()}
        total   = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        self.curriculum_weights = weights
        return weights

    # ------------------------------------------------------------------
    # Adapter recommendation
    # ------------------------------------------------------------------

    def _recommend_adapter(
        self,
        dominant_error: str,
        composite: float,
        error_counts: Dict[str, int],
    ) -> str:
        """
        Recommend OFT or BOFT for the next training round.

        BOFT (higher capacity) when model needs to learn new facts
        or rebuild broken reasoning chains.
        OFT  (geometry-preserving) when knowledge is intact but
        reasoning pathway or bias needs adjustment.
        """
        # Explicit BOFT triggers
        if dominant_error in BOFT_ERRORS:
            return "boft"

        # Majority vote from error distribution
        boft_votes = sum(
            count for err, count in error_counts.items()
            if err in BOFT_ERRORS
        )
        total_votes = sum(error_counts.values())
        if total_votes > 0 and boft_votes / total_votes > 0.4:
            return "boft"

        # Low composite score: upgrade to BOFT
        if composite < self.config.adapter_upgrade_threshold:
            return "boft"

        return "oft"

    # ------------------------------------------------------------------
    # Stagnation detection
    # ------------------------------------------------------------------

    def _is_stagnating(self) -> bool:
        w = self.config.stagnation_window
        if len(self._round_scores) < w:
            return False
        recent = self._round_scores[-w:]
        return (max(recent) - min(recent)) < self.config.stagnation_tolerance
