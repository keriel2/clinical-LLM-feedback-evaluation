# Evaluation of Critic-Driven Cognitive-Error Feedback for Clinical Language Model Adaptation

Code, per-question predictions and analysis scripts for the paper. All tables and figures can be regenerated from `results/` without retraining.

## Repository structure

| Folder | Contents |
|---|---|
| `src/` | Training pipeline: `trainer.py` (supervised, margin and DPO objectives), `train_one_condition.py` (conditions), `evaluate.py` |
| `src/agents/` | Task classifier, example selector, rule-based cognitive-error critic (`evaluation_agent.py`, complete decision rules), feedback updater, training coordinator |
| `scripts/` | `run_caise.sh` (train and evaluate one condition), queue and status helpers |
| `results/` | `eval_results.json` (accuracy and per-question predictions) and `compute_profile.json` for every reported run |
| `analysis/` | `paired_tests.py` (exact McNemar with Holm adjustment), table script |
| `make_figs.py` | Regenerates Figures 2 and 3 |

## Conditions

| Paper | `--condition` | Additional settings | Results folder |
|---|---|---|---|
| C0 | none | base model | `C0_base` |
| C1 | `static_sft` | batch 8 | `C1_static_s*` |
| C2 | `static_sft` | `BATCH=4` | `C2_static_batch4_s*` |
| C3 | `static_coord` | | `C3_loop_feedback_off_s*` |
| C4 | `quality_only` | | `C4_quality_selection_s*` |
| C5 | `error_curriculum` | | `C5_error_feedback_selection_s*` |
| C6 | `fmaca` | `CE_NEGATIVE_MODE=typed` | `C6_full_typed_s*` |
| C7 | `fmaca` | `CE_NEGATIVE_MODE=random` | `C7_full_random_negatives_s*` |
| C8 | `fmaca` | `CE_NEGATIVE_MODE=shuffled_types` (error types reassigned at random) | `C8_full_random_error_types_s*` |
| C9 | `fmaca` | `CONTRASTIVE_LOSS=dpo` | `C9_full_dpo_s42` |
| Sensitivity | as above | `MODEL_PATH=<Llama-3.1-8B-Instruct>`; `FMACA_ADAPTER=lora FMACA_LR=1e-4` | `llama_*`, `lora_*` |

In `eval_results.json`, the key `medqa` is the four-option MedQA test set and `biasmedqa` is the five-option MedQA test set (original questions, no added bias content).

## Reproducing the paper

```bash
# Train and evaluate one condition (seed 42, GPUs 0,1); models are read from $MODELS_DIR
MODELS_DIR=/path/to/models bash scripts/run_caise.sh static_coord 42 0,1 qwen

# Paired tests, e.g. error-feedback selection vs. feedback-off loop, seed 42
python analysis/paired_tests.py \
  C5=results/C5_error_feedback_selection_s42/eval_results.json \
  C3=results/C3_loop_feedback_off_s42/eval_results.json

# Figures
python make_figs.py
```

## Data

MedQA, MedMCQA and PubMedQA are publicly available and are not redistributed. `src/prepare_data.py` builds the 20,500-item training set; `results/training_item_ids.json` lists the exact items used.

## Requirements

Python 3.10, PyTorch, Transformers, PEFT, bitsandbytes. Experiments used 4-bit NF4 quantisation on 24 GB GPUs.
