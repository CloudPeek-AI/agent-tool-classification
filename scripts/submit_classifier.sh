#!/bin/bash
# Submit encoder-only classifier training on the HEC cluster for ONE OR MORE
# model presets at once, each as its own independent job.
#
#   scripts/submit_classifier.sh <model_preset> [<model_preset> ...]
#
# All jobs are submitted immediately; SLURM schedules them concurrently.
# Runs on gpu-short (12h). SLURM assigns the GPU automatically.
#
# Model presets:
#   bert        bert-base-uncased
#   roberta     roberta-base
#   electra     google/electra-base-discriminator
#   modernbert  answerdotai/ModernBERT-base
#   deberta     microsoft/deberta-v3-base           ← recommended
#   all         shorthand to submit all five presets
#
# Env overrides:
#   EPOCHS       max training epochs     (default: 15)
#   PATIENCE     early-stopping patience (default: 3)
#   LR           override learning rate  (default: per-preset)
#   BATCH_SIZE   override batch size     (default: per-preset)
#   LOGDIR       where SLURM .out/.err go (default: logs)
#   DRY_RUN=1    preview without submitting
#
# Examples:
#   scripts/submit_classifier.sh deberta
#   scripts/submit_classifier.sh all
#   DRY_RUN=1 scripts/submit_classifier.sh all

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, so src/* and scripts/* paths resolve

LOGDIR="${LOGDIR:-logs}"
EPOCHS="${EPOCHS:-15}"
PATIENCE="${PATIENCE:-3}"

MODELS=("$@")
[ "${#MODELS[@]}" -eq 0 ] && { echo "Usage: $0 <preset> [<preset>...]  (presets: bert roberta electra modernbert deberta all)" >&2; exit 1; }

# Expand 'all' shorthand.
expanded=()
for m in "${MODELS[@]}"; do
  if [ "$m" = "all" ]; then
    expanded+=(bert roberta electra modernbert deberta)
  else
    expanded+=("$m")
  fi
done
MODELS=("${expanded[@]}")

# --- Model presets -----------------------------------------------------------
# Sets: P_MODEL, P_LR, P_BATCH_SIZE, P_MEM (host RAM)
resolve_model() {
  case "$1" in
    bert)       P_MODEL="bert-base-uncased";                  P_LR="2e-5"; P_BATCH_SIZE=16; P_MEM=16G ;;
    roberta)    P_MODEL="roberta-base";                       P_LR="2e-5"; P_BATCH_SIZE=16; P_MEM=16G ;;
    electra)    P_MODEL="google/electra-base-discriminator";  P_LR="3e-5"; P_BATCH_SIZE=32; P_MEM=16G ;;
    modernbert) P_MODEL="answerdotai/ModernBERT-base";        P_LR="5e-5"; P_BATCH_SIZE=32; P_MEM=16G ;;
    deberta)    P_MODEL="microsoft/deberta-v3-base";          P_LR="1e-5"; P_BATCH_SIZE=16; P_MEM=24G ;;
    *) return 1 ;;
  esac
  return 0
}

submit_one() {
  local preset="$1"
  if ! resolve_model "$preset"; then
    echo "Unknown preset '$preset'. Choose: bert | roberta | electra | modernbert | deberta | all" >&2
    exit 1
  fi

  local model="${MODEL_OVERRIDE:-$P_MODEL}"
  local lr="${LR:-$P_LR}"
  local batch="${BATCH_SIZE:-$P_BATCH_SIZE}"
  local mem="${MEM:-$P_MEM}"
  local slug; slug=$(echo "$model" | sed 's#/#__#g')

  echo ">>> $preset | $model | lr=$lr | batch=$batch | mem=$mem"

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "    DRY_RUN: would submit -> outputs/${slug}_lr${lr}_bs${batch}_ep${EPOCHS}/"
    return 0
  fi

  local job_id
  job_id=$(sbatch --parsable \
    --job-name="classifier-${preset}" \
    --partition=gpu-short \
    --gres=gpu:1 \
    --time=12:00:00 \
    --mem="$mem" \
    --cpus-per-task=4 \
    --output="${LOGDIR}/classifier_${preset}_%j.out" \
    --error="${LOGDIR}/classifier_${preset}_%j.err" \
    --export=ALL,MODEL="$model",LR="$lr",BATCH_SIZE="$batch",EPOCHS="$EPOCHS",PATIENCE="$PATIENCE",LOGDIR="$LOGDIR" \
    scripts/run_classifier.slurm)

  echo "    submitted job $job_id -> outputs/${slug}_lr${lr}_bs${batch}_ep${EPOCHS}/"
}

mkdir -p "$LOGDIR"
echo "partition=gpu-short gres=gpu:1 time=12:00:00 | models: ${MODELS[*]} | logs: $LOGDIR/"
echo

for m in "${MODELS[@]}"; do
  submit_one "$m"
  echo
done

echo "all submissions complete (${#MODELS[@]} model(s)). SLURM will schedule them concurrently."
