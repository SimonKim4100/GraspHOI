#!/usr/bin/env bash
# clone and patch the third-party code, then create the conda environment; safe to re-run
#   CUDA_HOME=/path/to/cuda-12.8 bash tools/install.sh
# CUDA_HOME can be left out when the CUDA 12.8 nvcc is on PATH. GRASPHOI_ENV (environment name, default grasphoi)
# and MAX_JOBS (parallel compile jobs, default 8) can be set the same way.
set -eEo pipefail
trap 'echo "[FAIL] line $LINENO: $BASH_COMMAND" >&2' ERR

fail() { echo "[FAIL] $*" >&2; exit 1; }

if [ -n "$CUDA_HOME" ]; then
    CUDA=$(cd "$CUDA_HOME" 2>/dev/null && pwd) || fail "CUDA_HOME=$CUDA_HOME does not exist"
elif command -v nvcc >/dev/null; then
    CUDA=$(dirname "$(dirname "$(command -v nvcc)")")
fi
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT=$PWD
ENV_NAME=${GRASPHOI_ENV:-grasphoi}
[ "$ENV_NAME" != base ] || fail "use a separate environment, not base"

command -v git >/dev/null || fail "git not found"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found (NVIDIA driver)"
GPU_CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader) || fail "nvidia-smi cannot read the GPUs"
[ -n "$GPU_CAPS" ] || fail "no NVIDIA GPU found"
BLACKWELL=$(echo "$GPU_CAPS" | awk -F. '$1+0 >= 10 { b = 1 } END { if (b) print 1 }')
CONDA=${CONDA_EXE:-$(command -v conda || true)}
[ -x "$CONDA" ] || fail "conda not found"
{ [ -n "$CUDA" ] && [ -x "$CUDA/bin/nvcc" ]; } || fail "CUDA toolkit not found: set CUDA_HOME to a CUDA 12.8 install"
CUDA_VERSION=$("$CUDA/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
[ "${CUDA_VERSION%%.*}" = 12 ] || fail "the CUDA toolkit at $CUDA is CUDA $CUDA_VERSION; PyTorch here is built with CUDA 12.8"
[ -z "$BLACKWELL" ] || [ "${CUDA_VERSION#12.}" -ge 8 ] || fail "this GPU needs CUDA 12.8; the CUDA toolkit at $CUDA is CUDA $CUDA_VERSION"
echo "[ ok ] CUDA $CUDA_VERSION at $CUDA"

# third-party code
clone() {  # clone <name> <url> [patch ...]
    local dir="third_party/$1" url="$2" p
    shift 2
    if [ -d "$dir/.git" ] && git -C "$dir" rev-parse -q --verify HEAD >/dev/null; then
        echo "[skip] $dir (cloned)"
    else
        [ ! -e "$dir" ] || fail "$dir exists but is not a git clone; remove it and run again"
        git clone "$url" "$dir"
    fi
    for p in "$@"; do
        if git -C "$dir" apply --reverse --check "$ROOT/patches/$p" 2>/dev/null; then
            echo "[skip] $p (applied)"
        else
            git -C "$dir" apply "$ROOT/patches/$p"
            echo "[ ok ] $p"
        fi
    done
}
clone CameraHMR https://github.com/pixelite1201/CameraHMR
clone pixel-perfect-depth https://github.com/gangweix/pixel-perfect-depth pixel-perfect-depth-utils3d.patch
clone HOI-DETR https://github.com/AhmadDarKhalil/HOI-DETR hoi-detr-mmcv-without-compiled-ops.patch
clone sam-3d-body https://github.com/facebookresearch/sam-3d-body sam-3d-body-dinov3-hub.patch
clone sam-3d-objects https://github.com/facebookresearch/sam-3d-objects \
    sam-3d-objects-intrinsics.patch sam-3d-objects-dinov2-hub.patch sam-3d-objects-lean-imports.patch
clone detectron2 https://github.com/facebookresearch/detectron2 detectron2-no-pkg-resources.patch

# conda environment
eval "$("$CONDA" shell.bash hook)"
ENVS=$(conda env list) || fail "conda env list failed"
if awk '{print $1}' <<<"$ENVS" | grep -xF "$ENV_NAME" >/dev/null; then
    echo "[skip] conda environment $ENV_NAME (exists)"
else
    conda create -n "$ENV_NAME" -c conda-forge --override-channels python=3.11 pip -y
fi
conda env config vars set -n "$ENV_NAME" PYTHONNOUSERSITE=1 >/dev/null
conda activate "$ENV_NAME"
[ "$(command -v python)" = "$CONDA_PREFIX/bin/python" ] || fail "environment $ENV_NAME has no Python of its own"
python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))' || fail "environment $ENV_NAME does not have Python 3.11"

pip install --upgrade pip
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install --build-constraint tools/build-constraints.txt -r requirements.txt

# CUDA extensions, built from source
export CUDA_HOME="$CUDA" PATH="$CONDA_PREFIX/bin:$CUDA/bin:$PATH" LD_LIBRARY_PATH="$CUDA/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
TORCH_CUDA_ARCH_LIST=$(echo "$GPU_CAPS" | sort -u | paste -sd';' -)
export TORCH_CUDA_ARCH_LIST FORCE_CUDA=1 MAX_JOBS="${MAX_JOBS:-8}"
EXT=()
python -c 'import torch, nvdiffrast.torch' 2>/dev/null || EXT+=(git+https://github.com/NVlabs/nvdiffrast.git)
python -c 'import torch, pytorch3d._C' 2>/dev/null || EXT+=(git+https://github.com/facebookresearch/pytorch3d.git)
python -c 'import torch, detectron2._C' 2>/dev/null || EXT+=(./third_party/detectron2)
if [ ${#EXT[@]} -gt 0 ]; then
    pip install --no-build-isolation "${EXT[@]}"
else
    echo "[skip] nvdiffrast, pytorch3d, detectron2 (installed)"
fi

if python -c 'import torch, utils3d, moge' 2>/dev/null; then
    echo "[skip] utils3d, MoGe (installed)"
else
    pip install --no-deps git+https://github.com/EasternJournalist/utils3d.git git+https://github.com/microsoft/MoGe.git
fi

SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
# makes SAM 3D Objects importable
echo "$ROOT/third_party/sam-3d-objects" > "$SITE/sam3d_objects.pth"

# RTX 50 series and other Blackwell GPUs: xformers' FlashAttention-3 kernels fail there, so turn them off
if [ -n "$BLACKWELL" ]; then
    cat > "$SITE/xformers_no_fa3.py" <<'EOF'
import importlib.util, sys
class NoFA3:
    def find_spec(self, name, path=None, target=None):
        if name != "xformers.ops.fmha.dispatch":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        load = spec.loader.exec_module
        spec.loader.exec_module = lambda module: (load(module), module._set_use_fa3(False))
        return spec
sys.meta_path.insert(0, NoFA3())
EOF
    echo "import xformers_no_fa3" > "$SITE/xformers_no_fa3.pth"
    echo "[ ok ] xformers FlashAttention-3 turned off (Blackwell GPU)"
fi

PYOPENGL_PLATFORM=egl python -c "import torch, xformers.ops, nvdiffrast.torch, pytorch3d._C, detectron2._C, moge, utils3d, open3d, pyrender; pyrender.OffscreenRenderer(8, 8).delete(); assert torch.__version__ == '2.8.0+cu128', torch.__version__; print('[ ok ] torch', torch.__version__)"
echo "Done. Next: conda activate $ENV_NAME, then the model weights (README, step 2)."
