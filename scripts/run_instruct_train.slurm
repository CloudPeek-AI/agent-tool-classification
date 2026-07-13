#!/bin/bash
#SBATCH --job-name=instruct-train
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/instruct_train_%j.out
#SBATCH --error=logs/instruct_train_%j.err
# NOTE: --gres, --partition are supplied at submit time by submit_instruct.sh.

# ---------------------------------------------------------------------------
# Train one LLM with LoRA SFT on the instruct dataset.
# Env vars exported from submit_instruct.sh:
#
#   MODEL        HuggingFace model ID (required)
#   LR           learning rate            (default: 2e-4)
#   LORA_R       LoRA rank                (default: 16)
#   LORA_ALPHA   LoRA alpha               (default: 32)
#   BATCH_SIZE   per-device batch size    (default: 4)
#   GRAD_ACCUM   gradient accum steps     (default: 4)
#   EPOCHS       max training epochs      (default: 5)
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p "${LOGDIR:-logs}"

: "${MODEL:?MODEL not set — submit via scripts/submit_instruct.sh}"

source scripts/_env.sh
export WANDB_DISABLED=true

echo "model=${MODEL} | lr=${LR:-2e-4} | lora_r=${LORA_R:-16} | batch=${BATCH_SIZE:-4} | grad_accum=${GRAD_ACCUM:-4} | epochs=${EPOCHS:-5} | node=$(hostname) | gpu=${CUDA_VISIBLE_DEVICES:-?}"

python src/train_instruct.py \
    --model      "${MODEL}" \
    --lr         "${LR:-2e-4}" \
    --lora_r     "${LORA_R:-16}" \
    --lora_alpha "${LORA_ALPHA:-32}" \
    --batch_size "${BATCH_SIZE:-4}" \
    --grad_accum "${GRAD_ACCUM:-4}" \
    --epochs     "${EPOCHS:-5}"
