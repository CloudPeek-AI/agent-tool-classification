# Shared environment activation — SOURCED (not executed) by the SLURM jobs.
# Loads the cluster modules and activates the conda env created by setup_env.sh.
# Every value can be overridden from the environment; submit_vllm.sh forwards
# them via --export=ALL.

# UTF-8 locale — avoids conda/Python latin-1 UnicodeEncodeError on the cluster.
export LANG="${LANG:-C.UTF-8}"
export LC_ALL=C.UTF-8
export PYTHONUTF8=1

: "${MODULE_MINIFORGE:=miniforge/20251003}"
# Do NOT load a system CUDA module by default: vLLM's torch wheel bundles its own
# CUDA + NCCL, and a system libnccl on LD_LIBRARY_PATH shadows it (undefined
# symbol: ncclCommWindowDeregister). Only the GPU driver is needed. Set
# MODULE_CUDA=cuda/12.9 to override if you ever need the system toolkit.
: "${MODULE_CUDA:=}"
# FlashInfer JIT-compiles CUDA kernels with -std=c++20. If the default system GCC
# is too old to support C++20, nvcc silently drops the flag and libcu++ fails with
# "requires at least C++17". Load a newer GCC (>=11) to fix this.
# Override via env: MODULE_GCC=gcc/13.2 scripts/submit_vllm.sh coder-next
: "${MODULE_GCC:=}"
: "${TEACHER_ENV:=/storage/hpc/41/dolamull/envs/teacher}"   # conda env name or -p prefix path

# `source /etc/profile` makes the `module` command available in a non-interactive
# batch shell. These init scripts aren't `set -u` clean, so relax nounset around
# them. Conda is initialized via its shell hook below, then the prefix env is
# activated with `conda activate`.
set +u
source /etc/profile 2>/dev/null || true
module purge 2>/dev/null || true
[ -n "${MODULE_MINIFORGE}" ] && module load "${MODULE_MINIFORGE}"
[ -n "${MODULE_CUDA}" ]      && module load "${MODULE_CUDA}"
[ -n "${MODULE_GCC}" ]       && module load "${MODULE_GCC}"
# `conda activate` is unreliable in a non-interactive batch shell here, so run it
# best-effort AND prepend the env's bin to PATH — the latter guarantees the right
# python/packages regardless of whether conda activate took effect.
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${TEACHER_ENV}" 2>/dev/null || true
[[ "${TEACHER_ENV}" == /* && -d "${TEACHER_ENV}/bin" ]] && export PATH="${TEACHER_ENV}/bin:${PATH}"

# Fail fast (clear message) if python still isn't from the env.
if [[ "${TEACHER_ENV}" == /* ]] && [[ "$(command -v python)" != "${TEACHER_ENV}"/* ]]; then
  echo "ERROR: env not found at ${TEACHER_ENV} (python=$(command -v python)). Check TEACHER_ENV / that the env exists." >&2
  exit 1
fi

# Caches on scratch (fast, large), all under one XDG base. Override via env.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/hpc/41/dolamull/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" 2>/dev/null || true
export TOKENIZERS_PARALLELISM=false

# Load secrets (HF_TOKEN for gated models, etc.) from the gitignored repo .env.
# Jobs run with cwd = repo root, so ./.env resolves. Never commit .env.
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi
[ -n "${HF_TOKEN:-}" ] && export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

echo "env: $(python -V 2>&1) @ $(command -v python) | conda=${TEACHER_ENV} | cuda module=${MODULE_CUDA:-none} | hf_token=$([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)"
