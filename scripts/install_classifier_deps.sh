#!/bin/bash
# Install missing classifier dependencies into the existing teacher conda env.
# Run this once on the login node before submitting classifier jobs:
#
#   bash scripts/install_classifier_deps.sh

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

set +u
source /etc/profile 2>/dev/null || true
module purge 2>/dev/null || true
module load "${MODULE_MINIFORGE:-miniforge/20251003}"
eval "$(conda shell.bash hook)"

TEACHER_ENV="${TEACHER_ENV:-/storage/hpc/41/dolamull/envs/teacher}"
conda activate "${TEACHER_ENV}" 2>/dev/null || true
[[ "${TEACHER_ENV}" == /* && -d "${TEACHER_ENV}/bin" ]] && export PATH="${TEACHER_ENV}/bin:${PATH}"

echo "Installing into: $(command -v python) (${TEACHER_ENV})"

pip install --upgrade scikit-learn evaluate accelerate datasets trl peft bitsandbytes appdirs

echo "---"
python -c "import sklearn, evaluate, accelerate, datasets, trl, peft; print('OK | scikit-learn', sklearn.__version__, '| evaluate', evaluate.__version__, '| accelerate', accelerate.__version__, '| datasets', datasets.__version__, '| trl', trl.__version__, '| peft', peft.__version__)"
