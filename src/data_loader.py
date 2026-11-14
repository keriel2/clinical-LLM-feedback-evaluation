"""
data_loader.py
--------------
Unified data loader for FMACA experiments.

Handles four datasets in different formats and normalises them
into a single dict structure compatible with FMACA agents:

    {
        "input":  str,   # clinical question / case text
        "output": str,   # gold answer text
        "task":   str,   # clinical_diagnosis | medical_qa | clinical_reasoning
        "answer": str,   # letter or label of correct answer (A/B/C/yes/no/maybe)
        "source": str,   # dataset name
        "id":     str,   # unique example id
    }

Dataset paths (update BASE_DIR if needed):
    MedQA   : ./datasets/MedQA-USMLE-4-options/
    MedMCQA : ./datasets/medmcqa/
    PubMedQA: ./datasets/PubMedQA/pqal/
    BiasMedQA:./datasets/biasmedqa/

Usage:
    from data_loader import FMACADataLoader
    loader = FMACADataLoader()
    train_data = loader.load_all(split="train")
    test_data  = loader.load_all(split="test")
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Dataset paths
# -----------------------------------------------------------------------
BASE_DIR = "./datasets"

PATHS = {
    "medqa": {
        "train": f"{BASE_DIR}/MedQA-USMLE-4-options/phrases_no_exclude_train.jsonl",
        "test":  f"{BASE_DIR}/MedQA-USMLE-4-options/phrases_no_exclude_test.jsonl",
    },
    "medmcqa": {
        "train": f"{BASE_DIR}/medmcqa/train-00000-of-00001.parquet",
        "test":  f"{BASE_DIR}/medmcqa/test-00000-of-00001.parquet",
    },
    "pubmedqa": {
        "train": f"{BASE_DIR}/PubMedQA/pqal/pqal_train_dev_set.json",
        "test":  f"{BASE_DIR}/PubMedQA/pqal/pqal_test_set.json",
    },
    "biasmedqa": {
        "train": f"{BASE_DIR}/biasmedqa/train.jsonl",
        "test":  f"{BASE_DIR}/biasmedqa/test.jsonl",
    },
}

# Answer index mapping for MedMCQA (cop field: 0=A,1=B,2=C,3=D)
MEDMCQA_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
MEDMCQA_OPTION_FIELDS  = ["opa", "opb", "opc", "opd"]


# -----------------------------------------------------------------------
# Unified loader
# -----------------------------------------------------------------------

class FMACADataLoader:
    """
    Loads and normalises all four clinical datasets for FMACA.

    All examples are returned as dicts with keys:
        input, output, task, answer, source, id
    """

    def __init__(self, base_dir: str = BASE_DIR):
        self.base_dir = base_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(
        self,
        split: str = "train",
        datasets: Optional[List[str]] = None,
        max_per_dataset: Optional[int] = None,
    ) -> List[Dict]:
        """
        Load and combine examples from all four datasets.

        Args:
            split          : "train" or "test"
            datasets       : list of dataset names to include
                             (default: all four)
            max_per_dataset: cap examples per dataset (useful for pilot runs)

        Returns:
            List of normalised example dicts
        """
        if datasets is None:
            datasets = ["medqa", "medmcqa", "pubmedqa", "biasmedqa"]

        all_examples = []
        for name in datasets:
            try:
                examples = self._load_dataset(name, split)
                if max_per_dataset:
                    examples = examples[:max_per_dataset]
                all_examples.extend(examples)
                logger.info(
                    "Loaded %d examples from %s (%s)",
                    len(examples), name, split,
                )
            except Exception as e:
                logger.warning("Failed to load %s: %s", name, e)

        logger.info("Total examples loaded: %d", len(all_examples))
        return all_examples

    def load_dataset(
        self,
        name: str,
        split: str = "train",
        max_examples: Optional[int] = None,
    ) -> List[Dict]:
        """Load a single dataset by name."""
        examples = self._load_dataset(name, split)
        if max_examples:
            examples = examples[:max_examples]
        return examples

    def get_stats(self, examples: List[Dict]) -> Dict:
        """Return dataset statistics."""
        from collections import Counter
        return {
            "total":        len(examples),
            "by_source":    dict(Counter(e["source"] for e in examples)),
            "by_task":      dict(Counter(e["task"]   for e in examples)),
            "avg_input_len": sum(len(e["input"].split()) for e in examples)
                             / max(len(examples), 1),
        }

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_dataset(self, name: str, split: str) -> List[Dict]:
        loaders = {
            "medqa":     self._load_medqa,
            "medmcqa":   self._load_medmcqa,
            "pubmedqa":  self._load_pubmedqa,
            "biasmedqa": self._load_biasmedqa,
        }
        if name not in loaders:
            raise ValueError(f"Unknown dataset: {name}. "
                             f"Choose from {list(loaders.keys())}")
        return loaders[name](split)

    # ---- MedQA --------------------------------------------------------

    def _load_medqa(self, split: str) -> List[Dict]:
        """
        Format: JSONL
        Fields: question, options (dict), answer_idx, metamap_phrases
        Task mapping: clinical_diagnosis (USMLE Step 2/3) or medical_qa
        """
        path = PATHS["medqa"].get(split)
        if not path or not os.path.exists(path):
            # Fall back to test if train missing
            path = PATHS["medqa"]["test"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"MedQA file not found: {path}")

        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)

                question = raw.get("question", "").strip()
                options  = raw.get("options", {})
                answer_key = raw.get("answer_idx", "A")
                answer_text = options.get(answer_key, "")

                # Format options into readable string
                options_str = " | ".join(
                    f"{k}: {v}" for k, v in sorted(options.items())
                )
                input_text = (
                    f"{question}\n\nOptions: {options_str}"
                )

                # Task: diagnosis questions have clinical presentations
                task = self._infer_task_medqa(question)

                examples.append({
                    "input":  input_text,
                    "output": answer_text,
                    "task":   task,
                    "answer": answer_key,
                    "source": "medqa",
                    "id":     f"medqa_{split}_{i}",
                })

        return examples

    def _infer_task_medqa(self, question: str) -> str:
        """Infer task type from MedQA question text."""
        q_lower = question.lower()
        diagnosis_signals = [
            "presents with", "most likely diagnosis", "what is the diagnosis",
            "best diagnosis", "what does this patient have",
        ]
        reasoning_signals = [
            "mechanism", "pathophysiology", "why does", "explain",
            "what is the most likely cause", "what is the next step",
        ]
        if any(s in q_lower for s in diagnosis_signals):
            return "clinical_diagnosis"
        if any(s in q_lower for s in reasoning_signals):
            return "clinical_reasoning"
        return "medical_qa"

    # ---- MedMCQA -------------------------------------------------------

    def _load_medmcqa(self, split: str) -> List[Dict]:
        """
        Format: Parquet
        Fields: question, opa, opb, opc, opd, cop (int 0-3), explanation
        Task: medical_qa (Indian medical entrance exams)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pip install pandas pyarrow --break-system-packages")

        path = PATHS["medmcqa"].get(split)
        if not path or not os.path.exists(path):
            path = PATHS["medmcqa"]["test"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"MedMCQA file not found: {path}")

        df = pd.read_parquet(path)
        examples = []

        for i, row in df.iterrows():
            question = str(row.get("question", "")).strip()
            options  = {
                "A": str(row.get("opa", "")),
                "B": str(row.get("opb", "")),
                "C": str(row.get("opc", "")),
                "D": str(row.get("opd", "")),
            }
            cop = int(row.get("cop", 0))
            answer_key  = MEDMCQA_IDX_TO_LETTER.get(cop, "A")
            answer_text = options[answer_key]
            explanation = str(row.get("explanation", "")).strip()

            options_str = " | ".join(
                f"{k}: {v}" for k, v in options.items()
            )
            input_text = f"{question}\n\nOptions: {options_str}"

            # Use explanation as richer output if available
            output_text = (
                f"{answer_text}. {explanation}"
                if explanation and explanation != "nan"
                else answer_text
            )

            examples.append({
                "input":  input_text,
                "output": output_text,
                "task":   "medical_qa",
                "answer": answer_key,
                "source": "medmcqa",
                "id":     f"medmcqa_{split}_{i}",
            })

        return examples

    # ---- PubMedQA ------------------------------------------------------

    def _load_pubmedqa(self, split: str) -> List[Dict]:
        """
        Format: JSON (nested dict keyed by PubMed ID)
        Fields: QUESTION, CONTEXTS, LONG_ANSWER, final_decision (yes/no/maybe)
        Task: clinical_reasoning (evidence-based biomedical QA)

        Files:
            train: pqal_train_dev_set.json  (~449 examples)
            test:  pqal_test_set.json       (~500 examples)
        """
        json_file = PATHS["pubmedqa"].get(split)
        if not json_file or not os.path.exists(json_file):
            # Fallback: use test set
            json_file = PATHS["pubmedqa"]["test"]
        if not os.path.exists(json_file):
            raise FileNotFoundError(
                f"PubMedQA file not found: {json_file}"
            )

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = []
        for i, (pmid, record) in enumerate(data.items()):
            question     = record.get("QUESTION", "").strip()
            long_answer  = record.get("LONG_ANSWER", "").strip()
            decision     = record.get("final_decision", "").strip()
            contexts     = record.get("CONTEXTS", [])

            # Combine context paragraphs into input
            context_str = " ".join(contexts) if contexts else ""
            input_text  = (
                f"{question}\n\nContext: {context_str}"
                if context_str else question
            )

            # Output: long answer + decision
            output_text = (
                f"{long_answer} (Answer: {decision})"
                if long_answer else decision
            )

            examples.append({
                "input":  input_text,
                "output": output_text,
                "task":   "clinical_reasoning",
                "answer": decision,
                "source": "pubmedqa",
                "id":     f"pubmedqa_{pmid}",
            })

        return examples

    # ---- BiasMedQA -----------------------------------------------------

    def _load_biasmedqa(self, split: str) -> List[Dict]:
        """
        Format: JSONL (same structure as MedQA — derived from MedQA test set)
        Use test.jsonl for evaluation; train.jsonl for bias-aware training.
        Task: clinical_diagnosis (bias stress test)
        """
        path = PATHS["biasmedqa"].get(split)

        # Fallback: try phrases_ variants
        if not path or not os.path.exists(path):
            alt = f"{BASE_DIR}/biasmedqa/phrases_{split}.jsonl"
            if os.path.exists(alt):
                path = alt

        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"BiasMedQA file not found for split '{split}'. "
                f"Available: {os.listdir(BASE_DIR + '/biasmedqa')}"
            )

        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)

                question   = raw.get("question", "").strip()
                options    = raw.get("options", {})
                answer_key = raw.get("answer_idx",
                             raw.get("answer", "A"))
                answer_text = options.get(answer_key, "")

                options_str = " | ".join(
                    f"{k}: {v}" for k, v in sorted(options.items())
                )
                input_text = f"{question}\n\nOptions: {options_str}"

                examples.append({
                    "input":  input_text,
                    "output": answer_text,
                    "task":   "clinical_diagnosis",
                    "answer": answer_key,
                    "source": "biasmedqa",
                    "id":     f"biasmedqa_{split}_{i}",
                })

        return examples


# -----------------------------------------------------------------------
# Quick inspection utility
# -----------------------------------------------------------------------

def inspect_raw_files():
    """Print one raw record from each dataset for format verification."""
    print("\n=== MedQA (JSONL) ===")
    p = PATHS["medqa"]["test"]
    if os.path.exists(p):
        with open(p) as f:
            print(json.loads(f.readline()))

    print("\n=== MedMCQA (Parquet) ===")
    p = PATHS["medmcqa"]["test"]
    if os.path.exists(p):
        try:
            import pandas as pd
            df = pd.read_parquet(p)
            print(df.columns.tolist())
            print(df.iloc[0].to_dict())
        except ImportError:
            print("Install pandas: pip install pandas pyarrow --break-system-packages")

    print("\n=== PubMedQA (JSON) ===")
    pqal_dir = PATHS["pubmedqa"]["dir"]
    if os.path.isdir(pqal_dir):
        for fname in os.listdir(pqal_dir):
            fp = os.path.join(pqal_dir, fname)
            if fname.endswith(".json"):
                with open(fp) as f:
                    data = json.load(f)
                first_key = next(iter(data))
                print(f"File: {fname}")
                print(f"Keys in record: {list(data[first_key].keys())}")
                break
        else:
            print(f"Contents of {pqal_dir}:", os.listdir(pqal_dir))

    print("\n=== BiasMedQA (JSONL) ===")
    p = PATHS["biasmedqa"]["test"]
    if os.path.exists(p):
        with open(p) as f:
            print(json.loads(f.readline()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Inspecting raw formats ===")
    inspect_raw_files()

    print("\n=== Loading all datasets (test, max 5 each) ===")
    loader = FMACADataLoader()
    examples = loader.load_all(split="test", max_per_dataset=5)

    print(f"\nTotal loaded: {len(examples)}")
    print("\nStats:", loader.get_stats(examples))

    print("\nSample examples:")
    for ex in examples[:2]:
        print(f"\n[{ex['source']}] task={ex['task']}")
        print(f"  input:  {ex['input'][:120]}...")
        print(f"  output: {ex['output'][:80]}")
        print(f"  answer: {ex['answer']}")
