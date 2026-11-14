"""
taxonomy.py
-----------
Single source of truth for cognitive error taxonomy and curriculum categories.
Imported by EvaluationAgent, ExampleSelectionAgent, FeedbackAgent, AdapterAgent.

Cognitive error taxonomy grounded in clinical literature:
  Croskerry (2002) — cognitive error types in emergency medicine
  Graber et al. (2005) — diagnostic error in internal medicine
  Schiff et al. (2013) — contributory causes of diagnostic error
  NPJ Digital Medicine (2025) — cognitive bias in clinical LLMs

Curriculum categories grounded in clinical education:
  Reflect the range of case types needed for competency development.
"""

# -----------------------------------------------------------------------
# Cognitive error taxonomy
# All agents must use these constants — no local redefinitions.
# -----------------------------------------------------------------------

COGNITIVE_ERROR_TYPES = [
    "none",               # no significant error detected
    "premature_closure",  # stopped reasoning before full picture considered
    "anchoring",          # fixated on first clinical cue; ignored later evidence
    "knowledge_gap",      # reasoning structure present but factually wrong
    "reasoning_failure",  # facts present but inference chain broken
    "safety_omission",    # missed clinical red flag or safety-critical finding
]

# Errors that require higher-capacity adaptation (BOFT)
BOFT_ERRORS = {"knowledge_gap", "reasoning_failure"}

# Errors that require geometry-preserving adaptation (OFT)
OFT_ERRORS  = {"anchoring", "premature_closure", "safety_omission", "none"}

# -----------------------------------------------------------------------
# Curriculum categories
# -----------------------------------------------------------------------

CURRICULUM_CATEGORIES = [
    "differential_reasoning",  # cases requiring broad differential diagnosis
    "misleading_cue",          # cases with contradictory / late-breaking evidence
    "factual_evidence",        # factual / evidence-heavy questions
    "multi_step_reasoning",    # multi-step clinical reasoning chains
    "safety_critical",         # cases with red flags / safety-critical findings
    "routine",                 # standard cases (downweighted as model improves)
]

# Cognitive error → curriculum category to upweight
ERROR_TO_CURRICULUM = {
    "anchoring":         "misleading_cue",
    "premature_closure": "differential_reasoning",
    "knowledge_gap":     "factual_evidence",
    "reasoning_failure": "multi_step_reasoning",
    "safety_omission":   "safety_critical",
    "none":              "routine",
}

# Curriculum category → adapter mode
CURRICULUM_TO_ADAPTER = {
    "differential_reasoning": "oft",
    "misleading_cue":         "oft",
    "factual_evidence":       "boft",
    "multi_step_reasoning":   "boft",
    "safety_critical":        "oft",
    "routine":                "oft",
}

# Task types
TASK_TYPES = [
    "clinical_diagnosis",
    "medical_qa",
    "clinical_reasoning",
]

# Curriculum category per task type (for example tagging)
TASK_TO_CURRICULUM = {
    "clinical_diagnosis":  "differential_reasoning",
    "medical_qa":          "factual_evidence",
    "clinical_reasoning":  "multi_step_reasoning",
}

# Default curriculum weight distribution
DEFAULT_CURRICULUM_WEIGHTS = {cat: 1.0 / len(CURRICULUM_CATEGORIES)
                               for cat in CURRICULUM_CATEGORIES}

# Cognitive error -> adapter mode (used by AdapterAgent)
OFT  = "oft"
BOFT = "boft"

ERROR_TO_ADAPTER = {
    "anchoring":         OFT,
    "premature_closure": OFT,
    "knowledge_gap":     BOFT,
    "reasoning_failure": BOFT,
    "safety_omission":   OFT,
    "none":              OFT,
}


# -----------------------------------------------------------------------
# Stable example ID utility
# Replaces Python hash() which is not reproducible across runs.
# Uses SHA-256 truncated to 16 hex chars.
# -----------------------------------------------------------------------
import hashlib

def stable_example_id(text: str) -> str:
    """
    Stable, reproducible example ID from input text.
    Use instead of str(hash(...)) which varies across Python processes.
    Critical for per-example error history tracking across rounds.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
