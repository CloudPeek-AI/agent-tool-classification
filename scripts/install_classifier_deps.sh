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

# Remove the broken user-local wandb (uses np.float_ removed in NumPy 2.0)
# which shadows the conda env and crashes trl at import time.
pip uninstall -y wandb 2>/dev/null || true
# Reinstall a NumPy-2.0-compatible version into the conda env.
pip install --upgrade "wandb>=0.17.0"

pip install --upgrade scikit-learn evaluate accelerate datasets trl peft bitsandbytes

echo "---"
python -c "import sklearn, evaluate, accelerate, datasets, trl, peft, wandb; print('OK | scikit-learn', sklearn.__version__, '| evaluate', evaluate.__version__, '| accelerate', accelerate.__version__, '| datasets', datasets.__version__, '| trl', trl.__version__, '| peft', peft.__version__, '| wandb', wandb.__version__)"
