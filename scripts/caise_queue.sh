#!/bin/bash
# Runs the CAiSE jobs in priority order, each on the number of free GPUs it needs.
# A GPU counts as free when it uses < 1000 MiB and is not already given to a job of this queue.
# Usage: nohup bash caise_queue.sh > ablation_caise/queue.log 2>&1 &
cd .; mkdir -p ablation_caise
POOL=${POOL:-"1 2 3 4 5 6 7"}
Q=${MODELS_DIR}/Qwen2.5-7B-Instruct
L=${MODELS_DIR}/Llama-3.1-8B-Instruct
MI=${MODELS_DIR}/Mistral-7B-Instruct-v0.1
# ngpu | environment | condition | seed | tag            (highest priority first)
JOBS="
2|BATCH=4|static_sft|42|b4
2|BATCH=4|static_sft|43|b4
2|BATCH=4|static_sft|44|b4
4|CONTRASTIVE_LOSS=dpo|fmaca|42|dpo
2|MODEL_PATH=$L|static_sft|42|llama
2|MODEL_PATH=$L|static_coord|42|llama
2|MODEL_PATH=$L|error_curriculum|42|llama
4|MODEL_PATH=$L|fmaca|42|llama
2|FMACA_ADAPTER=lora FMACA_LR=1e-4|static_sft|42|lora
2|FMACA_ADAPTER=lora FMACA_LR=1e-4|static_coord|42|lora
4|FMACA_ADAPTER=lora FMACA_LR=1e-4|fmaca|42|lora
2|MODEL_PATH=$MI|static_sft|42|mistral
2|MODEL_PATH=$MI|static_coord|42|mistral
2|MODEL_PATH=$MI|error_curriculum|42|mistral
4|MODEL_PATH=$MI|fmaca|42|mistral
4|CONTRASTIVE_LOSS=dpo|fmaca|43|dpo
4|CONTRASTIVE_LOSS=dpo|fmaca|44|dpo
"
declare -A BUSY   # gpu -> pid
free_gpus() {
  for g in $POOL; do
    p=${BUSY[$g]}; if [ -n "$p" ] && kill -0 $p 2>/dev/null; then continue; fi; unset BUSY[$g]
    m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "$m" -lt 1000 ] && echo -n "$g "
  done; }
while IFS='|' read -r n envs cond seed tag; do
  [ -z "$n" ] && continue
  out=ablation_caise/${cond}_${tag}_s${seed}
  if [ -f $out/$cond/eval_results.json ]; then echo "[$(date '+%F %T')] skip $out (done)"; continue; fi
  while true; do
    set -- $(free_gpus); if [ $# -ge $n ]; then gpus=$(echo "$@" | tr ' ' '\n' | head -n $n | paste -sd,); break; fi
    sleep 300
  done
  mkdir -p $out
  env $envs bash run_caise.sh $cond $seed $gpus $tag > $out/run.nohup 2>&1 &
  pid=$!; for g in ${gpus//,/ }; do BUSY[$g]=$pid; done
  echo "[$(date '+%F %T')] launched $out on GPUs $gpus (pid $pid) env: $envs"
  sleep 240   # let it claim memory before the next check
done <<< "$JOBS"
wait; echo "[$(date '+%F %T')] queue done"
