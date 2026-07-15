#!/bin/bash
# Submit LoRA instruction-tuning + evaluation on the HEC cluster for ONE OR
# MORE model presets. Each model gets a training job followed by a dependent
# evaluation job (runs automatically after training completes).
#
#   scripts/submit_instruct.sh <model_preset> [<model_preset> ...]
#
# Model presets:
#   qwen7b      Qwen/Qwen2.5-7B-Instruct          ← recommended
#   qwen3b      Qwen/Qwen2.5-3B-Instruct
#   llama3b     meta-llama/Llama-3.2-3B-Instruct
#   llama1b     meta-llama/Llama-3.2-1B-Instruct
#   phi         microsoft/Phi-3.5-mini-instruct
#   mistral     mistralai/Mistral-7B-Instruct-v0.3
#   gemma       google/gemma-2-2b-it
#   all         shorthand to submit all seven presets
#
# Runs on gpu-short (12h train + 2h eval). SLURM assigns the GPU automatically.
#
# Env overrides:
#   EPOCHS      max training epochs   (default: 5)
#   LR          learning rate         (default: per-preset)
#   LORA_R      LoRA rank             (default: 16)
#   LORA_ALPHA  LoRA alpha            (default: 32)
#   BATCH_SIZE  per-device batch size (default: per-preset)
#   GRAD_ACCUM  gradient accum steps  (default: 4)
#   LOGDIR      log directory         (default: logs)
#   DRY_RUN=1   preview without submitting
#
# Examples:
#   scripts/submit_instruct.sh qwen7b
#   scripts/submit_instruct.sh all
#   EPOCHS=3 scripts/submit_instruct.sh llama3b phi
#   DRY_RUN=1 scripts/submit_instruct.sh all

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

LOGDIR="${LOGDIR:-logs}"
EPOCHS="${EPOCHS:-5}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
OUTPUTS_DIR="${OUTPUTS_DIR:-/scratch/hpc/41/dolamull/instruct_outputs}"

MODELS=("$@")
[ "${#MODELS[@]}" -eq 0 ] && { echo "Usage: $0 <preset> [<preset>...]  (presets: qwen7b qwen3b llama3b llama1b phi mistral gemma all)" >&2; exit 1; }

# Expand 'all' shorthand.
expanded=()
for m in "${MODELS[@]}"; do
  if [ "$m" = "all" ]; then
    expanded+=(qwen7b qwen3b llama3b llama1b phi mistral gemma)
  else
    expanded+=("$m")
  fi
done
MODELS=("${expanded[@]}")

# --- Model presets -----------------------------------------------------------
# Sets: P_MODEL, P_LR, P_BATCH_SIZE, P_MEM
resolve_model() {
  case "$1" in
    qwen7b)   P_MODEL="Qwen/Qwen2.5-7B-Instruct";                  P_LR="2e-4"; P_BATCH_SIZE=4; P_MEM=48G ;;
    qwen3b)   P_MODEL="Qwen/Qwen2.5-3B-Instruct";                  P_LR="2e-4"; P_BATCH_SIZE=8; P_MEM=32G ;;
    llama3b)  P_MODEL="meta-llama/Llama-3.2-3B-Instruct";          P_LR="2e-4"; P_BATCH_SIZE=8; P_MEM=32G ;;
    llama1b)  P_MODEL="meta-llama/Llama-3.2-1B-Instruct";          P_LR="3e-4"; P_BATCH_SIZE=16;P_MEM=16G ;;
    phi)      P_MODEL="microsoft/Phi-3.5-mini-instruct";            P_LR="2e-4"; P_BATCH_SIZE=8; P_MEM=32G ;;
    mistral)  P_MODEL="mistralai/Mistral-7B-Instruct-v0.3";        P_LR="2e-4"; P_BATCH_SIZE=4; P_MEM=48G ;;
    gemma)    P_MODEL="google/gemma-2-2b-it";                       P_LR="2e-4"; P_BATCH_SIZE=8; P_MEM=24G ;;
    *) return 1 ;;
  esac
  return 0
}

submit_one() {
  local preset="$1"
  if ! resolve_model "$preset"; then
    echo "Unknown preset '$preset'. Choose: qwen7b qwen3b llama3b llama1b phi mistral gemma all" >&2
    exit 1
  fi

  local model="${MODEL_OVERRIDE:-$P_MODEL}"
  local lr="${LR:-$P_LR}"
  local batch="${BATCH_SIZE:-$P_BATCH_SIZE}"
  local mem="${MEM:-$P_MEM}"
  local model_slug; model_slug=$(echo "$model" | sed 's#/#__#g')
  local merged_dir="${OUTPUTS_DIR}/${model_slug}_lr${lr}_r${LORA_R}_ep${EPOCHS}/merged"

  echo ">>> $preset | $model | lr=$lr | lora_r=$LORA_R | batch=$batch | grad_accum=$GRAD_ACCUM | mem=$mem"

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "    DRY_RUN: would submit train -> eval for $model"
    return 0
  fi

  # Submit training job.
  local train_id
  train_id=$(sbatch --parsable \
    --job-name="itrain-${preset}" \
    --partition=gpu-short \
    --gres=gpu:nvidia_h200_nvl:1 \
    --time=12:00:00 \
    --mem="$mem" \
    --cpus-per-task=4 \
    --output="${LOGDIR}/instruct_train_${preset}_%j.out" \
    --error="${LOGDIR}/instruct_train_${preset}_%j.err" \
    --export=ALL,MODEL="$model",LR="$lr",LORA_R="$LORA_R",LORA_ALPHA="$LORA_ALPHA",BATCH_SIZE="$batch",GRAD_ACCUM="$GRAD_ACCUM",EPOCHS="$EPOCHS",LOGDIR="$LOGDIR",OUTPUTS_DIR="$OUTPUTS_DIR" \
    scripts/run_instruct_train.slurm)
  echo "    training job: $train_id"

  # Submit eval job dependent on training completing successfully.
  local eval_id
  eval_id=$(sbatch --parsable \
    --dependency=afterok:"$train_id" \
    --kill-on-invalid-dep=yes \
    --job-name="ieval-${preset}" \
    --partition=gpu-short \
    --gres=gpu:nvidia_h200_nvl:1 \
    --time=02:00:00 \
    --mem="$mem" \
    --cpus-per-task=4 \
    --output="${LOGDIR}/instruct_eval_${preset}_%j.out" \
    --error="${LOGDIR}/instruct_eval_${preset}_%j.err" \
    --export=ALL,MERGED_DIR="$merged_dir",LOGDIR="$LOGDIR" \
    scripts/run_instruct_eval.slurm)
  echo "    eval job:     $eval_id (after $train_id)"
  echo "    results ->    ${merged_dir}/../eval_results.json"
}

mkdir -p "$LOGDIR"
echo "partition=gpu-short gres=gpu:nvidia_h200_nvl:1 | models: ${MODELS[*]} | logs: $LOGDIR/"
echo

for m in "${MODELS[@]}"; do
  submit_one "$m"
  echo
done

echo "all submissions complete (${#MODELS[@]} model(s), 2 jobs each). SLURM will schedule them concurrently."
