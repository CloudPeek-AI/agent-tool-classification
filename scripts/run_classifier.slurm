#!/bin/bash
#SBATCH --job-name=classifier-train
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/classifier_%j.out
#SBATCH --error=logs/classifier_%j.err
# NOTE: --gres, --partition, --time, --output/--error are supplied at submit
# time by scripts/submit_classifier.sh (they depend on the model / queue).
# The directives above are fallbacks for running this file directly.

# ---------------------------------------------------------------------------
# Worker: train one encoder-only classifier model.
# Driven by env vars exported from scripts/submit_classifier.sh:
#
#   MODEL        HuggingFace model ID (required)
#   LR           peak learning rate    (default: model preset)
#   BATCH_SIZE   per-device batch size (default: model preset)
#   EPOCHS       max training epochs   (default: 15)
#   PATIENCE     early-stopping epochs (default: 3)
#   MAX_LENGTH   max token length      (default: 256)
#   WEIGHT_DECAY AdamW weight decay    (default: 0.01)
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p "${LOGDIR:-logs}"

: "${MODEL:?MODEL not set — submit via scripts/submit_classifier.sh}"

source scripts/_env.sh
export WANDB_DISABLED=true

echo "model=${MODEL} | lr=${LR:-default} | batch=${BATCH_SIZE:-default} | epochs=${EPOCHS:-15} | node=$(hostname) | gpu=${CUDA_VISIBLE_DEVICES:-?}"

python src/train_classifier.py \
    --model      "${MODEL}" \
    --epochs     "${EPOCHS:-15}" \
    --patience   "${PATIENCE:-3}" \
    --max_length "${MAX_LENGTH:-256}" \
    --weight_decay "${WEIGHT_DECAY:-0.01}" \
    ${LR:+--lr          "${LR}"} \
    ${BATCH_SIZE:+--batch_size "${BATCH_SIZE}"}
