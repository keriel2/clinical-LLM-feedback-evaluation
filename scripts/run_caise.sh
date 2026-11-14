#!/bin/bash
# CAiSE baselines: one training run + evaluation, any model, any FMACA-CE condition.
# Usage: bash run_caise.sh <condition> <seed> <gpus> <tag>
#   condition: static_sft | static_coord | quality_only | error_curriculum | fmaca
# Environment (all optional):
#   MODEL_PATH        base model (default Qwen2.5-7B-Instruct)
#   BATCH             batch size (default 8)        EPOCHS (default 3)
#   CE_NEGATIVE_MODE  typed | random | shuffled_types (fmaca only, default typed)
#   CONTRASTIVE_LOSS  margin | dpo (fmaca only, default margin)
#   FMACA_ADAPTER     oft | lora (default oft)      FMACA_LR (default 2e-5)
# Output: ablation_caise/<condition>_<tag>_s<seed>/
set -e; set -o pipefail
COND=${1:?condition}; SEED=${2:?seed}; GPUS=${3:?gpus}; TAG=${4:?tag}
cd .
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL_PATH=${MODEL_PATH:-${MODELS_DIR}/Qwen2.5-7B-Instruct}
OUT=./ablation_caise/${COND}_${TAG}_s${SEED}; mkdir -p "$OUT"
echo "condition=$COND seed=$SEED gpus=$GPUS tag=$TAG model=$MODEL_PATH batch=${BATCH:-8} epochs=${EPOCHS:-3} ce=${CE_NEGATIVE_MODE:-typed} closs=${CONTRASTIVE_LOSS:-margin} adapter=${FMACA_ADAPTER:-oft} lr=${FMACA_LR:-2e-5} start=$(date)"
CE_NEGATIVE_MODE=${CE_NEGATIVE_MODE:-typed} CONTRASTIVE_LOSS=${CONTRASTIVE_LOSS:-margin} \
FMACA_ADAPTER=${FMACA_ADAPTER:-oft} FMACA_LR=${FMACA_LR:-2e-5} \
FMACA_SEED=$SEED PYTHONHASHSEED=$SEED CUDA_VISIBLE_DEVICES=$GPUS \
python -u train_one_condition.py --condition "$COND" --model_path "$MODEL_PATH" \
  --epochs ${EPOCHS:-3} --batch_size ${BATCH:-8} --max_per_dataset 10000 \
  --datasets medqa medmcqa pubmedqa --output_dir "$OUT" --seed "$SEED" 2>&1 | tee "$OUT/train.log"
CUDA_VISIBLE_DEVICES=${GPUS%%,*} python -u evaluate.py --model_path "$MODEL_PATH" \
  --adapter_path "$OUT/$COND/final_adapter" --datasets medqa biasmedqa pubmedqa \
  --max_examples 0 --split test --output_file "$OUT/$COND/eval_results.json" 2>&1 | tee "$OUT/eval.log"
echo "done=$(date)"
