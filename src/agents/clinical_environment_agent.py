"""
clinical_environment_agent.py
-----------------------------
FMACA-CLE: Clinical Learning Environment Agent.

This agent converts a static medical QA example into a staged clinical
reasoning environment:

1. Evidence extraction
2. Differential reasoning
3. Cognitive-bias check
4. Final-answer verification

It returns a stage-specific repair pair:

    rejected = model-generated flawed reasoning / answer
    chosen   = stage-specific gold-constrained repair target

This is NOT generic DPO. It is process-supervised, cognitive-error-aware
contrastive repair for clinical reasoning.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Dict, Optional, List


logger = logging.getLogger(__name__)


@dataclass
class ClinicalStageFeedback:
    weakest_stage: str
    error_type: str
    stage_prompt: str
    chosen_repair: str
    rejected_output: str
    evidence_score: float
    differential_score: float
    bias_score: float
    final_score: float
    should_train: bool
    reason: str


class ClinicalEnvironmentAgent:
    """
    Lightweight clinical learning environment.

    The goal is not to perfectly grade medicine.
    The goal is to create useful stage-specific repair supervision
    when the model output appears weak or critic-detected error exists.
    """

    STAGES = [
        "evidence_extraction",
        "differential_reasoning",
        "cognitive_bias_check",
        "final_answer",
    ]

    def __init__(
        self,
        min_output_words: int = 5,
        final_match_bonus: float = 0.35,
    ):
        self.min_output_words = min_output_words
        self.final_match_bonus = final_match_bonus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_stage_feedback(
        self,
        original_prompt: str,
        input_text: str,
        gold_output: str,
        model_output: str,
        critic_error_type: Optional[str] = None,
        example_id: Optional[str] = None,
    ) -> ClinicalStageFeedback:
        """
        Build process-supervision feedback for one example.

        If a critic error type is provided, it guides stage selection.
        Otherwise we use heuristic stage scores.
        """

        critic_error_type = self._normalize_error(critic_error_type)
        output = (model_output or "").strip()
        gold = (gold_output or "").strip()
        input_text = (input_text or "").strip()

        if not output or len(output.split()) < self.min_output_words:
            weakest = "final_answer"
            error = "incomplete_answer"
            scores = {
                "evidence_extraction": 0.0,
                "differential_reasoning": 0.0,
                "cognitive_bias_check": 0.0,
                "final_answer": 0.0,
            }
        else:
            scores = self._score_stages(
                input_text=input_text,
                gold_output=gold,
                model_output=output,
            )

            weakest = min(scores, key=scores.get)
            error = critic_error_type if critic_error_type != "none" else self._stage_to_error(weakest)

        # If the critic explicitly detects a cognitive error, override stage
        # with a clinically meaningful stage.
        if critic_error_type in {
            "anchoring",
            "premature_closure",
            "misleading_cue",
            "cognitive_bias",
        }:
            weakest = "cognitive_bias_check"
            error = critic_error_type

        elif critic_error_type in {
            "reasoning_failure",
            "multi_step_reasoning",
            "differential_error",
        }:
            weakest = "differential_reasoning"
            error = critic_error_type

        elif critic_error_type in {
            "evidence_omission",
            "factual_evidence",
            "knowledge_gap",
        }:
            weakest = "evidence_extraction"
            error = critic_error_type

        elif critic_error_type in {
            "safety_omission",
            "safety_critical",
        }:
            weakest = "cognitive_bias_check"
            error = "safety_omission"

        # If the answer is already very close to gold and no critic error,
        # avoid adding noisy pairs.
        final_score = scores.get("final_answer", 0.0)
        if critic_error_type == "none" and final_score >= 0.85:
            return ClinicalStageFeedback(
                weakest_stage=weakest,
                error_type="none",
                stage_prompt=original_prompt,
                chosen_repair=gold,
                rejected_output=output,
                evidence_score=scores.get("evidence_extraction", 0.0),
                differential_score=scores.get("differential_reasoning", 0.0),
                bias_score=scores.get("cognitive_bias_check", 0.0),
                final_score=final_score,
                should_train=False,
                reason="output appears close to gold; no repair pair added",
            )

        stage_prompt = self._make_stage_prompt(
            original_prompt=original_prompt,
            stage=weakest,
        )

        chosen = self._make_stage_repair_target(
            stage=weakest,
            input_text=input_text,
            gold_output=gold,
            critic_error_type=error,
        )

        reason = (
            f"weakest_stage={weakest}; critic_error={critic_error_type}; "
            f"scores={scores}"
        )

        return ClinicalStageFeedback(
            weakest_stage=weakest,
            error_type=error,
            stage_prompt=stage_prompt,
            chosen_repair=chosen,
            rejected_output=output,
            evidence_score=scores.get("evidence_extraction", 0.0),
            differential_score=scores.get("differential_reasoning", 0.0),
            bias_score=scores.get("cognitive_bias_check", 0.0),
            final_score=scores.get("final_answer", 0.0),
            should_train=True,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Stage scoring
    # ------------------------------------------------------------------

    def _score_stages(
        self,
        input_text: str,
        gold_output: str,
        model_output: str,
    ) -> Dict[str, float]:
        out = self._norm(model_output)
        gold = self._norm(gold_output)
        inp = self._norm(input_text)

        gold_terms = self._content_terms(gold)
        input_terms = self._content_terms(inp)

        # Use a small set of answer-relevant input cues.
        salient_input_terms = input_terms[:30]

        evidence_overlap = self._overlap_score(out, salient_input_terms)
        gold_overlap = self._overlap_score(out, gold_terms)

        reasoning_markers = [
            "because", "therefore", "suggest", "consistent", "due to",
            "rule out", "differential", "alternative", "however",
            "whereas", "most likely", "less likely"
        ]
        reasoning_score = min(
            1.0,
            sum(1 for m in reasoning_markers if m in out) / 3.0,
        )

        bias_markers = [
            "although", "however", "alternative", "rule out",
            "must also consider", "not simply", "despite"
        ]
        bias_score = min(
            1.0,
            sum(1 for m in bias_markers if m in out) / 2.0,
        )

        final_score = gold_overlap
        if self._answer_label(gold_output) and self._answer_label(gold_output) in model_output:
            final_score = min(1.0, final_score + self.final_match_bonus)

        return {
            "evidence_extraction": evidence_overlap,
            "differential_reasoning": 0.5 * reasoning_score + 0.5 * gold_overlap,
            "cognitive_bias_check": bias_score if final_score < 0.75 else max(bias_score, 0.75),
            "final_answer": final_score,
        }

    # ------------------------------------------------------------------
    # Repair target construction
    # ------------------------------------------------------------------

    def _make_stage_prompt(self, original_prompt: str, stage: str) -> str:
        if stage == "evidence_extraction":
            instruction = (
                "Clinical reasoning stage: EVIDENCE EXTRACTION.\n"
                "Identify the key clinical evidence that should guide the answer. "
                "Do not jump directly to the final answer."
            )

        elif stage == "differential_reasoning":
            instruction = (
                "Clinical reasoning stage: DIFFERENTIAL REASONING.\n"
                "Use the evidence to compare the most plausible diagnosis or answer "
                "against alternatives before giving the final answer."
            )

        elif stage == "cognitive_bias_check":
            instruction = (
                "Clinical reasoning stage: COGNITIVE BIAS CHECK.\n"
                "Check for anchoring, premature closure, misleading cues, and omitted "
                "safety-critical evidence before committing to the answer."
            )

        else:
            instruction = (
                "Clinical reasoning stage: FINAL ANSWER VERIFICATION.\n"
                "Provide the final answer and a concise justification grounded in the case."
            )

        return original_prompt.rstrip() + "\n\n" + instruction

    def _make_stage_repair_target(
        self,
        stage: str,
        input_text: str,
        gold_output: str,
        critic_error_type: str,
    ) -> str:
        gold = gold_output.strip()

        if stage == "evidence_extraction":
            return (
                "Key evidence should be extracted before answering. "
                "The answer must be grounded in the clinically relevant findings from the case. "
                f"Gold-constrained target answer: {gold}"
            )

        if stage == "differential_reasoning":
            return (
                "The reasoning should compare plausible alternatives and select the option "
                "best supported by the evidence. Avoid jumping from a single cue to a conclusion. "
                f"Gold-constrained target answer: {gold}"
            )

        if stage == "cognitive_bias_check":
            return (
                f"The earlier response reflects a possible {critic_error_type}. "
                "Repair the reasoning by checking for anchoring, premature closure, omitted evidence, "
                "and safety-critical considerations before answering. "
                f"Gold-constrained target answer: {gold}"
            )

        return (
            "The final answer should match the gold clinical target and include a concise justification. "
            f"Gold-constrained target answer: {gold}"
        )

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------

    def _normalize_error(self, error: Optional[str]) -> str:
        if not error:
            return "none"
        e = str(error).strip().lower()
        aliases = {
            "safety_critical": "safety_omission",
            "factual_error": "knowledge_gap",
            "evidence_missing": "evidence_omission",
            "none": "none",
            "": "none",
        }
        return aliases.get(e, e)

    def _stage_to_error(self, stage: str) -> str:
        return {
            "evidence_extraction": "evidence_omission",
            "differential_reasoning": "reasoning_failure",
            "cognitive_bias_check": "anchoring",
            "final_answer": "incorrect_final_answer",
        }.get(stage, "reasoning_failure")

    def _norm(self, text: str) -> str:
        text = str(text or "").lower()
        text = re.sub(r"[^a-z0-9\s]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _content_terms(self, text: str) -> List[str]:
        stop = {
            "the", "and", "for", "with", "that", "this", "from", "have",
            "has", "are", "was", "were", "will", "would", "could", "should",
            "patient", "patients", "answer", "question", "following", "most",
            "likely", "which", "what", "when", "where", "why", "how", "into",
            "than", "then", "them", "they", "their", "there", "these", "those",
            "because", "therefore", "option", "choice"
        }
        words = [w for w in text.split() if len(w) >= 4 and w not in stop]
        seen = set()
        unique = []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        return unique

    def _overlap_score(self, text: str, terms: List[str]) -> float:
        if not terms:
            return 0.0
        text_terms = set(text.split())
        hits = sum(1 for t in terms if t in text_terms)
        return min(1.0, hits / max(1, min(len(terms), 8)))

    def _answer_label(self, text: str) -> Optional[str]:
        m = re.search(r"\b([A-E])\b", str(text or ""))
        if m:
            return m.group(1)
        return None
