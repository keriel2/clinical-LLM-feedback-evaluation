"""
Adapter Agent
-------------
Dynamically selects between OFT and BOFT fine-tuning configurations
based on the Critic Agent's cognitive error signal each training round.

Motivation (from supervisor's direction):
  - LoRA uses additive low-rank updates: risks catastrophic forgetting
    of foundational clinical knowledge between tasks.
  - OFT (Orthogonal Fine-Tuning, Liu et al. NeurIPS 2023) preserves
    hyperspherical energy of pretrained weights — adapts without
    destroying pretrained geometry. Better for cases where knowledge
    is intact but reasoning pathway needs adjustment.
  - BOFT (Butterfly OFT) uses butterfly matrix factorization to reduce
    OFT parameter complexity from O(n²) to O(n log n). Better for cases
    where the model needs higher capacity adaptation (knowledge gaps).

Error-to-adapter mapping (clinically grounded):
  knowledge_gap       -> BOFT (high capacity: model needs new facts)
  anchoring           -> OFT  (preserve knowledge, adjust reasoning bias)
  premature_closure   -> OFT  (preserve knowledge, extend reasoning)
  availability_bias   -> BOFT (needs broader factual grounding)
  reasoning_failure   -> BOFT (high rank: inference chain needs rebuilding)
  instruction_mismatch-> OFT  (knowledge fine, only format needs adjusting)
  none / clean        -> OFT  (minimal intervention: preserve what works)

References:
  Liu et al. (2023). Orthogonal Fine-tuning via Butterfly Factorization. NeurIPS.
  Büyükakyüz (2024). BOFT extensions.
  HuggingFace PEFT library: OFTConfig, BOFTConfig.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter

from agents.taxonomy import BOFT_ERRORS, OFT_ERRORS, ERROR_TO_ADAPTER

logger = logging.getLogger(__name__)


# Adapter mode constants
OFT  = "oft"
BOFT = "boft"


@dataclass
class OFTParams:
    """Configuration for standard OFT."""
    r: int = 8                          # orthogonal block size
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    module_dropout: float = 0.0
    init_weights: bool = True
    coft: bool = False                  # constrained OFT variant


@dataclass
class BOFTParams:
    """Configuration for Butterfly OFT."""
    boft_block_size: int = 8
    boft_n_butterfly_factor: int = 2    # butterfly decomposition depth
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    boft_dropout: float = 0.1
    bias: str = "none"


@dataclass
class AdapterDecision:
    """Record of one adapter selection decision."""
    round_num: int
    task: str
    dominant_error: str
    adapter_mode: str
    oft_params: Optional[OFTParams]
    boft_params: Optional[BOFTParams]
    reason: str


class AdapterAgent:
    """
    Critic-driven OFT/BOFT adapter selection agent.

    Receives the dominant cognitive error type from the Feedback Agent
    and selects the appropriate fine-tuning configuration for the next round.

    Design:
      - Does not perform fine-tuning itself (that is the trainer's job)
      - Returns a config object the Coordinator passes to the trainer
      - Maintains decision history for ablation analysis
      - Supports per-task adapter decisions (diagnosis may need different
        adapter than reasoning)
    """

    def __init__(
        self,
        oft_params: Optional[OFTParams] = None,
        boft_params: Optional[BOFTParams] = None,
        seed: int = 42,
    ):
        self.oft_params  = oft_params  or OFTParams()
        self.boft_params = boft_params or BOFTParams()

        self._history: List[AdapterDecision] = []
        self._current_mode: Dict[str, str] = {}  # task -> current adapter mode

        logger.info(
            "AdapterAgent initialized | default OFT r=%d | BOFT block=%d butterfly=%d",
            getattr(self.oft_params, "oft_block_size", self.oft_params.r),
            self.boft_params.boft_block_size,
            self.boft_params.boft_n_butterfly_factor,
        )

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def select(
        self,
        dominant_error: str,
        task: str,
        round_num: int,
        error_history: Optional[List[str]] = None,
    ) -> Tuple[str, object]:
        """
        Select OFT or BOFT based on dominant cognitive error type.

        Uses TASK-SPECIFIC error history only.
        Does not let another task's errors influence this task's adapter.

        Args:
            dominant_error : most frequent non-none error for THIS task this round
            task           : clinical task type
            round_num      : current training round
            error_history  : error history for THIS TASK ONLY (not global)

        Returns:
            (adapter_mode: str, config: OFTParams | BOFTParams)
        """
        # BOFT disabled: shape conflict with 4-bit NF4 in multi-round training
        # All selections use OFT for v1 experiments
        if dominant_error in BOFT_ERRORS:
            adapter_mode = OFT

        # Rule 2: current error is non-serious → check task-specific history
        elif error_history:
            # Use only this task's recent history, excluding "none"
            task_recent = [e for e in error_history[-5:] if e != "none"]
            if task_recent:
                votes = Counter(ERROR_TO_ADAPTER.get(e, OFT) for e in task_recent)
                adapter_mode = OFT  # forced OFT for v1
            else:
                adapter_mode = OFT  # clean task history → OFT

        # Rule 3: no history or current error is "none" → OFT
        else:
            adapter_mode = ERROR_TO_ADAPTER.get(dominant_error, OFT)

        # Select config
        if adapter_mode == BOFT:
            config = self.boft_params
            reason = (
                f"error='{dominant_error}' requires higher capacity adaptation "
                f"(BOFT: butterfly O(n log n), block_size={self.boft_params.boft_block_size})"
            )
        else:
            config = self.oft_params
            reason = (
                f"error='{dominant_error}' requires geometry-preserving adaptation "
                f"(OFT: r={self.oft_params.r}, preserves hyperspherical energy)"
            )

        # Log decision
        decision = AdapterDecision(
            round_num=round_num,
            task=task,
            dominant_error=dominant_error,
            adapter_mode=adapter_mode,
            oft_params=self.oft_params if adapter_mode == OFT else None,
            boft_params=self.boft_params if adapter_mode == BOFT else None,
            reason=reason,
        )
        self._history.append(decision)
        self._current_mode[task] = adapter_mode

        logger.info(
            "Round %d | %s | AdapterAgent -> %s | %s",
            round_num, task, adapter_mode.upper(), reason,
        )

        return adapter_mode, config

    def get_peft_config(
        self,
        dominant_error: str,
        task: str,
        round_num: int,
        error_history: Optional[List[str]] = None,
    ):
        """
        Return a HuggingFace PEFT config object ready to pass to get_peft_model().

        Requires: pip install peft
        Falls back to returning a plain dict if PEFT not installed
        (for unit testing without GPU environment).
        """
        adapter_mode, params = self.select(
            dominant_error, task, round_num, error_history
        )

        try:
            from peft import OFTConfig, BOFTConfig

            if adapter_mode == BOFT:
                return BOFTConfig(
                    boft_block_size=params.boft_block_size,
                    boft_n_butterfly_factor=params.boft_n_butterfly_factor,
                    target_modules=params.target_modules,
                    boft_dropout=params.boft_dropout,
                    bias=params.bias,
                )
            else:
                return OFTConfig(
                    r=params.r,
                    target_modules=params.target_modules,
                    module_dropout=params.module_dropout,
                    init_weights=params.init_weights,
                    coft=params.coft,
                )

        except ImportError:
            logger.warning(
                "PEFT library not installed. Returning config dict for %s.",
                adapter_mode.upper(),
            )
            if adapter_mode == BOFT:
                return {
                    "type": "boft",
                    "boft_block_size": params.boft_block_size,
                    "boft_n_butterfly_factor": params.boft_n_butterfly_factor,
                    "target_modules": params.target_modules,
                    "boft_dropout": params.boft_dropout,
                }
            else:
                return {
                    "type": "oft",
                    "r": params.r,
                    "target_modules": params.target_modules,
                    "module_dropout": params.module_dropout,
                    "coft": params.coft,
                }

    def get_current_mode(self, task: str) -> str:
        """Return the currently active adapter mode for a task."""
        return self._current_mode.get(task, OFT)

    def get_history(self) -> List[Dict]:
        return [
            {
                "round": d.round_num,
                "task": d.task,
                "dominant_error": d.dominant_error,
                "adapter_mode": d.adapter_mode,
                "reason": d.reason,
            }
            for d in self._history
        ]

    def get_mode_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Return OFT vs BOFT usage counts per task.
        Useful for paper result tables.
        """
        distribution: Dict[str, Dict[str, int]] = {}
        for d in self._history:
            if d.task not in distribution:
                distribution[d.task] = {OFT: 0, BOFT: 0}
            distribution[d.task][d.adapter_mode] += 1
        return distribution

    def get_stats(self) -> Dict:
        total = len(self._history)
        oft_count  = sum(1 for d in self._history if d.adapter_mode == OFT)
        boft_count = sum(1 for d in self._history if d.adapter_mode == BOFT)
        return {
            "total_decisions": total,
            "oft_selections":  oft_count,
            "boft_selections": boft_count,
            "oft_fraction":  oft_count  / max(total, 1),
            "boft_fraction": boft_count / max(total, 1),
            "per_task_mode": self.get_mode_distribution(),
            "error_to_adapter_map": ERROR_TO_ADAPTER,
        }


class StaticAdapterAgent:
    """
    Baseline: always uses OFT regardless of error signal.
    Used in ablation conditions without dynamic adapter selection.
    """

    def __init__(self, oft_params: Optional[OFTParams] = None):
        self.oft_params = oft_params or OFTParams()
        self._calls = 0

    def select(self, dominant_error, task, round_num, error_history=None):
        self._calls += 1
        return OFT, self.oft_params

    def get_peft_config(self, dominant_error, task, round_num, error_history=None):
        try:
            from peft import OFTConfig
            return OFTConfig(
                r=getattr(self.oft_params, "oft_block_size", self.oft_params.r),
                target_modules=self.oft_params.target_modules,
                module_dropout=self.oft_params.module_dropout,
            )
        except ImportError:
            return {"type": "oft", "r": self.oft_params.r}

    def get_current_mode(self, task: str) -> str:
        return OFT

    def get_stats(self) -> Dict:
        return {"mode": "static_oft", "total_calls": self._calls}
