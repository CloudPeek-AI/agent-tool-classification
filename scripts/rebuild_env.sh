#!/bin/bash
# Tear down the corrupted teacher conda env and rebuild it from scratch.
# Run on the login node (no GPU needed):
#
#   bash scripts/rebuild_env.sh
#
# What it does:
#   1. Clears stale Cargo cache from home dir (quota fix)
#   2. Removes the existing prefix env
#   3. Creates a fresh Python 3.11 env at the same path
#   4. Installs PyTorch (CUDA 12.1 bundled wheel) + all project deps

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

TEACHER_ENV="${TEACHER_ENV:-/storage/hpc/41/dolamull/envs/teacher}"
MINIFORGE_MODULE="${MODULE_MINIFORGE:-miniforge/20251003}"
PYTHON_VERSION="3.11"
SCRATCH_CACHE="/scratch/hpc/41/dolamull/.cache"

# ── 1. Load conda ──────────────────────────────────────────────────────────────
set +u
source /etc/profile 2>/dev/null || true
module purge 2>/dev/null || true
module load "${MINIFORGE_MODULE}"
eval "$(conda shell.bash hook)"
set -u

echo "========================================================================"
echo "  Rebuilding conda env at: ${TEACHER_ENV}"
echo "  Python                 : ${PYTHON_VERSION}"
echo "========================================================================"

# ── 2. Free home-dir quota: remove stale Cargo cache (caused quota failure) ───
STALE_CARGO="${HOME}/.cache/puccinialin/cargo"
if [ -d "${STALE_CARGO}" ]; then
    echo "[1/5] Removing stale cargo cache from home (~/.cache/puccinialin/cargo)..."
    rm -rf "${STALE_CARGO}"
else
    echo "[1/5] No stale cargo cache found in home, skipping."
fi

# ── 3. Remove old env ──────────────────────────────────────────────────────────
if [ -d "${TEACHER_ENV}" ]; then
    echo "[2/5] Removing corrupted env..."
    conda deactivate 2>/dev/null || true
    conda env remove -p "${TEACHER_ENV}" -y
else
    echo "[2/5] No existing env found at ${TEACHER_ENV}, skipping removal."
fi

# ── 4. Create fresh env ────────────────────────────────────────────────────────
echo "[3/5] Creating fresh env (Python ${PYTHON_VERSION})..."
conda create -p "${TEACHER_ENV}" python="${PYTHON_VERSION}" -y

export PATH="${TEACHER_ENV}/bin:${PATH}"
conda activate "${TEACHER_ENV}" 2>/dev/null || true

echo "      Python: $(python --version) @ $(command -v python)"

# ── 5. Install PyTorch (bundles CUDA 12.1 — no system cuda module needed) ─────
echo "[4/5] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ── 6. Install all project dependencies ───────────────────────────────────────
echo "[5/5] Installing project dependencies..."

# Redirect Cargo/Rust build caches to scratch so they don't exhaust the
# home-directory quota. vllm pulls in llguidance which builds with maturin/Rust.
export CARGO_HOME="${CARGO_HOME:-${SCRATCH_CACHE}/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-${SCRATCH_CACHE}/rustup}"
mkdir -p "${CARGO_HOME}" "${RUSTUP_HOME}"
echo "      CARGO_HOME=${CARGO_HOME}"

pip install \
    "transformers>=4.40.0" \
    "datasets>=2.18.0" \
    "accelerate>=0.28.0" \
    "scikit-learn>=1.3.0" \
    "evaluate>=0.4.0" \
    "numpy>=1.24.0" \
    "huggingface-hub>=0.24" \
    "wandb>=0.17.0" \
    "trl" \
    "peft" \
    "bitsandbytes" \
    "vllm>=0.15.0"

# ── Smoke test ─────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "  Smoke test"
echo "========================================================================"
python - <<'EOF'
import torch, transformers, datasets, accelerate, sklearn, evaluate, trl, peft, wandb, bitsandbytes, vllm, huggingface_hub
print(f"torch           {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
print(f"transformers    {transformers.__version__}")
print(f"datasets        {datasets.__version__}")
print(f"accelerate      {accelerate.__version__}")
print(f"scikit-learn    {sklearn.__version__}")
print(f"evaluate        {evaluate.__version__}")
print(f"trl             {trl.__version__}")
print(f"peft            {peft.__version__}")
print(f"wandb           {wandb.__version__}")
print(f"bitsandbytes    {bitsandbytes.__version__}")
print(f"vllm            {vllm.__version__}")
print(f"huggingface-hub {huggingface_hub.__version__}")
print("ALL OK")
EOF

echo ""
echo "Done. Env rebuilt at ${TEACHER_ENV}"
echo "Activate with:  conda activate ${TEACHER_ENV}"
