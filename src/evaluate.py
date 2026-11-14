"""
evaluate.py
-----------
Evaluates a trained FMACA model (or baseline) on MedQA / BiasMedQA test sets.
Reports accuracy (exact match on MCQ answer letter).

Usage:
    # After training a condition, run:
    python evaluate.py \
        --model_path ${MODELS_DIR}/Qwen2.5-7B-Instruct \
        --adapter_path ./ablation_pilot/fmaca/adapter_round_3_oft \
        --datasets medqa biasmedqa \
        --output_file ./ablation_pilot/fmaca/eval_results.json

    # For baseline (no adapter):
    python evaluate.py \
        --model_path ${MODELS_DIR}/Qwen2.5-7B-Instruct \
        --datasets medqa biasmedqa \
        --output_file ./ablation_pilot/static_sft/eval_results.json
"""

import os
import json
import argparse
import logging
import sys
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, '.')
from data_loader import FMACADataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("evaluate")

MODEL_PATH = "${MODELS_DIR}/Qwen2.5-7B-Instruct"

PROMPT_TEMPLATE = """You are a clinical expert answering medical exam questions.
Read the question carefully and select the single best answer.
Respond with ONLY the answer letter (A, B, C, D, or E). Nothing else.

Question: {question}

Options:
{options}

Answer:"""


def load_model(model_path: str, adapter_path: Optional[str] = None):
    """Load model with optional PEFT adapter."""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading model in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = True

    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        logger.info("Loading adapter from %s ...", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info("Adapter loaded.")
    else:
        logger.info("No adapter — evaluating base model.")

    model.eval()
    return model, tokenizer


def format_prompt(example: Dict) -> str:
    """Format a MedQA example into an evaluation prompt."""
    question = example.get("question", example.get("input", ""))
    options  = example.get("options", {})

    # If options not in example dict, parse from input field
    if not options and "\nOptions:" in question:
        parts = question.split("\nOptions:")
        question = parts[0].strip()
        for opt_str in parts[1].strip().split(" | "):
            if ": " in opt_str:
                k, v = opt_str.split(": ", 1)
                options[k.strip()] = v.strip()

    options_str = "\n".join(
        f"{k}. {v}" for k, v in sorted(options.items())
    )
    return PROMPT_TEMPLATE.format(
        question=question,
        options=options_str,
    )


def extract_answer(generated_text: str) -> str:
    """Extract single letter answer from model output."""
    import re
    text = generated_text.strip().upper()
    # Pattern 1: "The correct answer is X" or "Answer: X"
    m = re.search(r'(?:ANSWER\s*(?:IS)?|CORRECT\s+ANSWER\s*(?:IS)?)\s*:?\s*([A-E])', text)
    if m: return m.group(1)
    # Pattern 2: "X." or "X)" at start
    m = re.match(r'^([A-E])[.):\s]', text)
    if m: return m.group(1)
    # Pattern 3: first valid letter
    for char in text:
        if char in "ABCD":
            return char
    return "?"


def extract_answer_pubmedqa(generated_text: str) -> str:
    """PubMedQA-specific: handles yes/no/maybe word forms from non-Qwen models."""
    import re
    text = generated_text.strip().upper()
    # Word-form first (Mistral/Llama output yes/no/maybe directly)
    m = re.match(r'^(YES|NO|MAYBE)', text)
    if m:
        return {"YES": "A", "NO": "B", "MAYBE": "C"}.get(m.group(1), "?")
    # Letter form A/B/C
    m = re.match(r'^([A-C])[.):\s]', text)
    if m: return m.group(1)
    for char in text:
        if char in "ABC":
            return char
    return "?"



def evaluate_dataset(
    model,
    tokenizer,
    examples: List[Dict],
    max_examples: Optional[int] = None,
    batch_size: int = 4,
    max_new_tokens: int = 5,
    dataset_name: str = "",
) -> Dict:
    """Run evaluation on a list of examples."""
    if max_examples:
        examples = examples[:max_examples]

    correct = 0
    total   = 0
    results = []

    logger.info("Evaluating %d examples...", len(examples))

    for i, ex in enumerate(examples):
        # Format prompt
        prompt = format_prompt(ex)

        # Get gold answer
        gold = ex.get("answer_idx", ex.get("answer", "?")).strip().upper()
        if len(gold) > 1:
            # Extract letter if full text given
            gold = gold[0] if gold else "?"

        # Generate
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768,
        ).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=5,   # only need one letter
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        pred_text  = tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred       = extract_answer_pubmedqa(pred_text) if dataset_name == "pubmedqa" else extract_answer(pred_text)

        is_correct = (pred == gold)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "id":        ex.get("id", f"ex_{i}"),
            "gold":      gold,
            "predicted": pred,
            "correct":   is_correct,
            "generated_text": pred_text.strip(),
        })

        if (i + 1) % 20 == 0:
            logger.info(
                "Progress: %d/%d | accuracy so far: %.1f%%",
                i + 1, len(examples), 100 * correct / total,
            )

    accuracy = correct / max(total, 1)
    logger.info("Final accuracy: %.1f%% (%d/%d)", 100 * accuracy, correct, total)

    return {
        "accuracy":    accuracy,
        "correct":     correct,
        "total":       total,
        "per_example": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   default=MODEL_PATH)
    parser.add_argument("--adapter_path", default=None,
                        help="Path to saved PEFT adapter (optional)")
    parser.add_argument("--datasets",     nargs="+",
                        default=["medqa"],
                        choices=["medqa", "medmcqa", "pubmedqa", "biasmedqa"])
    parser.add_argument("--max_examples", type=int, default=200,
                        help="Max test examples per dataset")
    parser.add_argument("--output_file",  default="./eval_results.json")
    parser.add_argument("--split",        default="test")
    parser.add_argument("--fulltext",     action="store_true",
                        help="Generate full reasoning text (max_new_tokens=128)")
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model_path, args.adapter_path)

    # Load test data
    loader   = FMACADataLoader()
    all_results = {}

    for dataset_name in args.datasets:
        logger.info("=== Evaluating on %s (%s) ===", dataset_name, args.split)
        examples = loader.load_dataset(dataset_name, split=args.split)

        # Reload raw examples with options dict for proper evaluation
        # (data_loader merges options into input string — we need them separate)
        raw_examples = _load_raw_examples(dataset_name, args.split)

        result = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            examples=raw_examples,
            max_examples=args.max_examples,
            dataset_name=dataset_name,
            max_new_tokens=128 if args.fulltext else 5,
        )
        all_results[dataset_name] = {
            "accuracy":    result["accuracy"],
            "correct":     result["correct"],
            "total":       result["total"],
            "per_example": result["per_example"],
        }
        logger.info("%s accuracy: %.1f%%", dataset_name, 100 * result["accuracy"])

    # Save
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("Results saved to %s", args.output_file)
    print("\n=== EVALUATION SUMMARY ===")
    for ds, res in all_results.items():
        print(f"  {ds}: {res['accuracy']*100:.1f}% ({res['correct']}/{res['total']})")


def _load_raw_examples(dataset_name: str, split: str) -> List[Dict]:
    """Load raw examples with options dict preserved."""
    import json

    BASE = "./datasets"

    if dataset_name in ("medqa", "biasmedqa"):
        path_map = {
            "medqa": {
                "train": f"{BASE}/MedQA-USMLE-4-options/phrases_no_exclude_train.jsonl",
                "test":  f"{BASE}/MedQA-USMLE-4-options/phrases_no_exclude_test.jsonl",
            },
            "biasmedqa": {
                "train": f"{BASE}/biasmedqa/train.jsonl",
                "test":  f"{BASE}/biasmedqa/test.jsonl",
            },
        }
        path = path_map[dataset_name].get(split, path_map[dataset_name]["test"])
        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    elif dataset_name == "medmcqa":
        import pandas as pd
        path = f"{BASE}/medmcqa/{'train' if split == 'train' else 'test'}-00000-of-00001.parquet"
        df = pd.read_parquet(path)
        examples = []
        for _, row in df.iterrows():
            cop = int(row.get("cop", -1))
            if cop == -1:
                continue
            idx_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            examples.append({
                "question":   str(row["question"]),
                "options":    {"A": str(row["opa"]), "B": str(row["opb"]),
                               "C": str(row["opc"]), "D": str(row["opd"])},
                "answer_idx": idx_map.get(cop, "A"),
            })
        return examples

    elif dataset_name == "pubmedqa":
        path = f"{BASE}/PubMedQA/pqal/pqal_test_set.json"
        with open(path) as f:
            data = json.load(f)
        examples = []
        # Map yes/no/maybe -> A/B/C so the existing letter-based
        # format_prompt()/extract_answer() pipeline works unchanged.
        decision_to_letter = {"yes": "A", "no": "B", "maybe": "C"}
        for pmid, record in data.items():
            decision = record.get("final_decision", "").strip().lower()
            letter   = decision_to_letter.get(decision, "?")
            if letter == "?":
                continue  # skip malformed records
            examples.append({
                "question":   record.get("QUESTION", ""),
                "options":    {"A": "yes", "B": "no", "C": "maybe"},
                "answer_idx": letter,
            })
        return examples

    return []


if __name__ == "__main__":
    main()
