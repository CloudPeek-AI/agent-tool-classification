#!/bin/bash
#SBATCH --job-name=instruct-eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/instruct_eval_%j.out
#SBATCH --error=logs/instruct_eval_%j.err
# NOTE: --gres, --partition are supplied at submit time by submit_instruct.sh.

# ---------------------------------------------------------------------------
# Evaluate the merged model on the instruct test set using vLLM.
# Env vars exported from submit_instruct.sh:
#
#   MERGED_DIR   path to the merged model directory (required)
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p "${LOGDIR:-logs}"

: "${MERGED_DIR:?MERGED_DIR not set — submit via scripts/submit_instruct.sh}"

source scripts/_env.sh

echo "merged_dir=${MERGED_DIR} | node=$(hostname) | gpu=${CUDA_VISIBLE_DEVICES:-?}"

python src/evaluate_instruct.py \
    --merged_dir "${MERGED_DIR}"
