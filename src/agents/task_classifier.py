"""
Task Classifier
---------------
Replaces RouterAgent. Identifies clinical task type from input text
and passes the label to downstream agents.

This is intentionally simple — it is a label function, not a policy.
No instruction functions. No routing probabilities.
No arbitrary choices that reviewers can attack.

Three task types derived from standard clinical NLP benchmarks:
  clinical_diagnosis  : identifying disease/condition from presentation
  medical_qa          : answering factual clinical questions
  clinical_reasoning  : multi-step reasoning through a clinical case

Classification uses keyword heuristics in v1.
Can be replaced with a fine-tuned classifier in v2.
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

TASK_TYPES = ["clinical_diagnosis", "medical_qa", "clinical_reasoning"]

# Keyword signals per task type
_DIAGNOSIS_KEYWORDS = [
    "diagnosis", "most likely", "what is the cause", "presents with",
    "differential", "condition", "disease", "syndrome", "disorder",
    "what does the patient have", "what is wrong",
]
_REASONING_KEYWORDS = [
    "walk through", "step by step", "reasoning", "explain your thinking",
    "how would you approach", "what is your thought process",
    "work through", "clinical reasoning",
]
_QA_KEYWORDS = [
    "what is", "what are", "how does", "mechanism", "treatment for",
    "first-line", "management of", "define", "which drug", "dosage",
    "side effect", "contraindication", "indication",
]


class TaskClassifier:
    """
    Lightweight task classifier.

    Identifies whether a training example is clinical_diagnosis,
    medical_qa, or clinical_reasoning based on input text signals.

    Used by Coordinator to:
      1. Label examples for per-task tracking
      2. Inform AdapterAgent of task context
      3. Inform SelectionAgent curriculum rebalancing
    """

    def __init__(self):
        self._counts: Dict[str, int] = {t: 0 for t in TASK_TYPES}
        logger.info("TaskClassifier initialized.")

    def classify(self, example: Dict) -> str:
        """
        Classify a single example into a task type.

        Falls back to the example's 'task' field if already labeled.
        Uses keyword heuristics otherwise.

        Returns one of TASK_TYPES.
        """
        # Use existing label if present and valid
        existing = example.get("task", "")
        if existing in TASK_TYPES:
            self._counts[existing] += 1
            return existing

        text = (example.get("input", "") + " " + example.get("output", "")).lower()

        # Score each task type
        scores = {
            "clinical_reasoning": sum(1 for kw in _REASONING_KEYWORDS if kw in text),
            "clinical_diagnosis": sum(1 for kw in _DIAGNOSIS_KEYWORDS if kw in text),
            "medical_qa":         sum(1 for kw in _QA_KEYWORDS if kw in text),
        }

        task = max(scores, key=scores.get)

        # Fallback if no keywords matched
        if scores[task] == 0:
            task = "medical_qa"

        self._counts[task] += 1
        return task

    def classify_batch(self, examples: List[Dict]) -> List[str]:
        return [self.classify(ex) for ex in examples]

    def get_stats(self) -> Dict:
        total = sum(self._counts.values())
        return {
            "total_classified": total,
            "per_task": dict(self._counts),
            "per_task_fraction": {
                t: c / max(total, 1)
                for t, c in self._counts.items()
            },
        }
