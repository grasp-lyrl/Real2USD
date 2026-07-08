#!/usr/bin/env bash
# ============================================================================
# Set up the Meta sam-3d-objects environment for the Real2USD SAM3D worker.
#
# Target: this desktop's RTX 5090 (Blackwell / sm_120). The upstream CUDA-12.1
# flow (environments/default.yml + cu121 wheels) does NOT run on Blackwell, so
# this uses the repo's "installation with 5090" recipe: torch 2.8.0 cu128 +
# sam3d-objects-single.yml, plus from-source pytorch3d. kaolin is intentionally
# skipped -- the fork is patched to shim its only runtime use (check_tensor) and
# make its notebook-viz import optional (see sam3d_objects/utils/kaolin_compat.py
# and the try/except in notebook/inference.py).
#
# Idempotent-ish: safe to re-run. Requires conda/mamba (miniforge). If you don't
# have it:  https://github.com/conda-forge/miniforge  (installs to ~/miniforge3).
#
# Overridable via env vars: SAM3D_ENV, SAM3D_REPO, MAMBA_ROOT_PREFIX.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM3D_REPO="${SAM3D_REPO:-$SCRIPT_DIR/sam-3d-objects}"
SAM3D_ENV="${SAM3D_ENV:-sam3d-objects}"
: "${MAMBA_ROOT_PREFIX:=$HOME/miniforge3}"
export MAMBA_ROOT_PREFIX

if [[ ! -d "$SAM3D_REPO/notebook" ]]; then
  echo "ERROR: sam-3d-objects clone not found at $SAM3D_REPO (needs notebook/, checkpoints/)." >&2
  echo "Clone the fork there, or set SAM3D_REPO=/path/to/sam-3d-objects." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$MAMBA_ROOT_PREFIX/etc/profile.d/conda.sh"
# shellcheck disable=SC1091
source "$MAMBA_ROOT_PREFIX/etc/profile.d/mamba.sh"

cd "$SAM3D_REPO"

echo "==== [1/6] create env '$SAM3D_ENV' (python 3.11) ===="
if ! conda env list | grep -qE "^\s*$SAM3D_ENV\s"; then
  mamba create -n "$SAM3D_ENV" python=3.11 -y
fi
conda activate "$SAM3D_ENV"

echo "==== [2/6] torch 2.8.0 cu128 (Blackwell / sm_120) ===="
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; assert torch.cuda.is_available(); \
  print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"

echo "==== [3/6] runtime deps (sam3d-objects-single.yml: spconv-cu120, gsplat, MoGe, ...) ===="
conda env update -f sam3d-objects-single.yml
# per README: refresh the env after the update
conda deactivate
conda activate "$SAM3D_ENV"

echo "==== [4/6] CUDA dev headers for compiling pytorch3d (cudart + cccl 12.8) ===="
mamba install -n "$SAM3D_ENV" -y -c nvidia -c conda-forge "cuda-cudart-dev=12.8.*" "cuda-cccl=12.8.*"

echo "==== [5/6] build pytorch3d from source against torch 2.8/cu128 ===="
export CUDA_HOME="$CONDA_PREFIX"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0"   # sm_120
export MAX_JOBS="$(nproc)"
pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"

echo "==== [6/6] install sam3d_objects package (no deps; single.yml provides them) ===="
pip install -e . --no-deps

echo "==== verify: worker import path ===="
# NOTE: import notebook/inference.py FIRST -- it sets LIDRA_SKIP_INIT before importing
# sam3d_objects (the public release ships no sam3d_objects.init submodule). A bare
# `import sam3d_objects` without that flag fails; the worker never does that.
python - <<'PY'
import sys; sys.path.insert(0, "notebook")
import pytorch3d  # noqa
from inference import Inference, load_image, load_single_mask  # noqa
print("OK: sam3d-objects env ready for the Real2USD worker.")
PY

cat <<EOF

Next: download the gated checkpoint (needs HF access to facebook/sam-3d-objects).
  $MAMBA_ROOT_PREFIX/bin/hf auth login          # paste your token
  TAG=hf
  hf download --repo-type model --local-dir checkpoints/\${TAG}-download --max-workers 1 facebook/sam-3d-objects
  mv checkpoints/\${TAG}-download/checkpoints checkpoints/\${TAG}
  rm -rf checkpoints/\${TAG}-download

Then run the worker:
  conda activate $SAM3D_ENV
  python scripts_sam3d_worker/run_sam3d_worker.py \\
    --queue-dir ~/Data/datasets/sam3d_queue --sam3d-repo $SAM3D_REPO
EOF
