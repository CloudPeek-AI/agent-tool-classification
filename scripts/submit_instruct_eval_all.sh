#!/bin/bash
# Submit evaluation jobs for all trained instruct models found in outputs/instruct/*/merged.
#
#   scripts/submit_instruct_eval_all.sh
#
# One eval job is submitted per merged model directory found.
# DRY_RUN=1 prints what would be submitted without actually submitting.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

LOGDIR="${LOGDIR:-logs}"
mkdir -p "$LOGDIR"

shopt -s nullglob
SCRATCH_OUTPUTS="${SCRATCH_OUTPUTS:-/scratch/hpc/41/dolamull/instruct_outputs}"
merged_dirs=(outputs/instruct/*/merged "${SCRATCH_OUTPUTS}"/*/merged)

# Warn about run dirs that exist but have no merged/ subdir (e.g. training failed before merge).
for run_dir in outputs/instruct/*/ "${SCRATCH_OUTPUTS}"/*/; do
  [ -d "$run_dir" ] || continue
  if [ ! -d "${run_dir}merged" ]; then
    echo "WARNING: no merged/ dir found in $run_dir — skipping (training may not have completed)" >&2
  fi
done

if [ "${#merged_dirs[@]}" -eq 0 ]; then
  echo "No merged model directories found under outputs/instruct/*/merged or ${SCRATCH_OUTPUTS}/*/merged" >&2
  exit 1
fi

echo "Found ${#merged_dirs[@]} model(s):"
for merged in "${merged_dirs[@]}"; do
  echo "  $merged"
done
echo

for merged in "${merged_dirs[@]}"; do
  # Resolve to absolute path — scratch paths are already absolute, local ones are relative.
  if [[ "$merged" = /* ]]; then
    abs_merged="$merged"
  else
    abs_merged="$PWD/$merged"
  fi
  run_name="$(basename "$(dirname "$merged")")"

  echo ">>> $run_name"

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "    DRY_RUN: would submit eval for $abs_merged"
    echo
    continue
  fi

  job_id=$(sbatch --parsable \
    --job-name="ieval-${run_name}" \
    --partition=gpu-short \
    --gres=gpu:nvidia_h200_nvl:1 \
    --time=02:00:00 \
    --mem=48G \
    --cpus-per-task=4 \
    --output="${LOGDIR}/instruct_eval_${run_name}_%j.out" \
    --error="${LOGDIR}/instruct_eval_${run_name}_%j.err" \
    --export=ALL,MERGED_DIR="${abs_merged}",LOGDIR="${LOGDIR}" \
    scripts/run_instruct_eval.slurm)

  echo "    submitted job $job_id -> ${merged}/../eval_results.json"
  echo
done

echo "all eval jobs submitted."
