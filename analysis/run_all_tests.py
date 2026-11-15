"""Reproduces every paired test reported in the paper, with the comparison families defined explicitly.

Usage (from the repository root):  python analysis/run_all_tests.py

Families and Holm adjustment:
  RQ1   C3 vs C1, C2 vs C1, C2 vs C3                     (per-seed tests, reported unadjusted)
  RQ2   C4 vs C3 and C5 vs C3 (the 18 selection tests)   (reported unadjusted; Holm shown for completeness)
  MAIN  the 69 component tests of the main model, Holm-adjusted as one family:
        per seed: C5 vs C4, C6 vs C5, C7 vs C6, C8 vs C6, C6 vs C3, C7 vs C3, C8 vs C3  (7 x 3 benchmarks x 3 seeds = 63)
        seed 42:  C9 vs C6, C9 vs C3                                                     (2 x 3 benchmarks = 6)
  SENS-A  sensitivity analysis (12 tests): Llama C3 vs C1, C5 vs C3, C6 vs C3; LoRA C3 vs C1
  SENS-B  LoRA full system (6 tests): LoRA C6 vs C3, C6 vs C1
  SENS-C  Mistral, if present: C3 vs C1, C5 vs C3
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from paired_tests import compare_family, print_rows

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
NAMES = {"C1": "C1_static", "C2": "C2_static_batch4", "C3": "C3_loop_feedback_off",
         "C4": "C4_quality_selection", "C5": "C5_error_feedback_selection", "C6": "C6_full_typed",
         "C7": "C7_full_random_negatives", "C8": "C8_full_random_error_types", "C9": "C9_full_dpo"}
SEEDS = (42, 43, 44)

def p(cond, seed):
    return os.path.join(R, f"{NAMES[cond]}_s{seed}", "eval_results.json")

def pair(a, b, seed):
    return (f"{a}_s{seed}", p(a, seed), f"{b}_s{seed}", p(b, seed))

def sens(name):
    return os.path.join(R, name, "eval_results.json")

print_rows(compare_family([pair(a, b, s) for s in SEEDS for a, b in (("C3", "C1"), ("C2", "C1"), ("C2", "C3"))]),
           "RQ1: execution controls (report unadjusted p)")
print_rows(compare_family([pair(a, "C3", s) for s in SEEDS for a in ("C4", "C5")]),
           "RQ2: selection vs. feedback-off loop (report unadjusted p)")

print_rows(compare_family([pair(a, "C1", s) for s in SEEDS for a in ("C5", "C6")]),
           "C5 and C6 vs conventional fine-tuning (RQ1, RQ2; Holm within these 18)")

main = [pair(a, b, s) for s in SEEDS for a, b in
        (("C5", "C4"), ("C6", "C5"), ("C7", "C6"), ("C8", "C6"), ("C6", "C3"), ("C7", "C3"), ("C8", "C3"))]
main += [pair("C9", "C6", 42), pair("C9", "C3", 42)]
print_rows(compare_family(main), "MAIN: component contrasts, Holm across the family")

print_rows(compare_family([
    ("L_C3", sens("llama_C3_s42"), "L_C1", sens("llama_C1_s42")),
    ("L_C5", sens("llama_C5_s42"), "L_C3", sens("llama_C3_s42")),
    ("L_C6", sens("llama_C6_s42"), "L_C3", sens("llama_C3_s42")),
    ("Lo_C3", sens("lora_C3_s42"), "Lo_C1", sens("lora_C1_s42"))]), "SENS-A: Llama and LoRA, Holm across the family")
print_rows(compare_family([
    ("Lo_C6", sens("lora_C6_s42"), "Lo_C3", sens("lora_C3_s42")),
    ("Lo_C6", sens("lora_C6_s42"), "Lo_C1", sens("lora_C1_s42"))]), "SENS-B: LoRA full system, Holm across the family")
mis = [("M_C3", "mistral_static_coord_s42", "M_C1", "mistral_static_sft_s42"),
       ("M_C5", "mistral_error_curriculum_s42", "M_C3", "mistral_static_coord_s42"),
       ("M_C6", "mistral_fmaca_s42", "M_C3", "mistral_static_coord_s42")]
mis = [(la, sens(a), lb, sens(b)) for la, a, lb, b in mis if os.path.exists(sens(a)) and os.path.exists(sens(b))]
if mis:
    print_rows(compare_family(mis), "SENS-C: Mistral, Holm across the family")

r = lambda n: os.path.join(R, n, "eval_results.json")
print_rows(compare_family([
    ("L_C5", r("llama_C5_s42"), "L_C1", r("llama_C1_s42")),
    ("L_C6", r("llama_C6_s42"), "L_C1", r("llama_C1_s42")),
    ("M_C5", r("mistral_error_curriculum_s42"), "M_C1", r("mistral_static_sft_s42")),
    ("M_C6", r("mistral_fmaca_s42"), "M_C1", r("mistral_static_sft_s42")),
    ("Lo_C6", r("lora_C6_s42"), "Lo_C1", r("lora_C1_s42"))]),
    "SENS-D: feedback conditions vs C1, sensitivity models (Holm within these 15)")
