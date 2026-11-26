#!/usr/bin/env bash
# download the no-account model weights; safe to re-run

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAILED=()

get() {
    local dest="$1" url="$2"
    if [ -s "$dest" ]; then echo "[skip] $dest (exists)"; return 0; fi
    mkdir -p "$(dirname "$dest")"
    echo "[get ] $dest"
    if wget -q --show-progress -c -O "$dest.part" "$url" && mv "$dest.part" "$dest"; then
        return 0
    fi
    rm -f "$dest.part"; FAILED+=("$dest"); return 1
}

gget() {  # gget <dest> <google-drive-file-id>
    local dest="$1" id="$2"
    if [ -s "$dest" ]; then echo "[skip] $dest (exists)"; return 0; fi
    if ! command -v gdown >/dev/null; then
        echo "[FAIL] $dest -- gdown not installed (pip install gdown)"; FAILED+=("$dest"); return 1
    fi
    mkdir -p "$(dirname "$dest")"
    echo "[get ] $dest (Google Drive)"
    if ! gdown "$id" -O "$dest"; then rm -f "$dest"; FAILED+=("$dest"); return 1; fi
}

# WiLoR (hand pose)
WILOR_HF="https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main"
get data/wilor_data/wilor_ckpts/checkpoints/wilor_final.ckpt "$WILOR_HF/pretrained_models/wilor_final.ckpt"
get data/wilor_data/dataset_config.yaml "$WILOR_HF/pretrained_models/dataset_config.yaml"
get data/wilor_data/mano/mano_mean_params.npz "$WILOR_HF/mano_data/mano_mean_params.npz"
if [ ! -s data/wilor_data/model_config.yaml ] &&
   get data/wilor_data/model_config.yaml "$WILOR_HF/pretrained_models/model_config.yaml"; then
    # repoint the MANO entries
    sed -i -e 's|^  DATA_DIR:.*|  DATA_DIR: data/wilor_data/mano|' \
           -e 's|^  MODEL_PATH:.*|  MODEL_PATH: data/wilor_data/mano|' \
           -e 's|^  MEAN_PARAMS:.*|  MEAN_PARAMS: data/wilor_data/mano/mano_mean_params.npz|' \
           data/wilor_data/model_config.yaml
fi

# Detectron2 ViTDet person detector
get data/camerahmr/model_final_f05665.pkl \
    "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"

# FoundationPose scorer + refiner
FP_W=object_pose/foundationpose_runtime/weights
if [ -s "$FP_W/2023-10-28-18-33-37/model_best.pth" ] && [ -s "$FP_W/2024-01-11-20-02-45/model_best.pth" ]; then
    echo "[skip] $FP_W (exists)"
elif command -v gdown >/dev/null; then
    echo "[get ] FoundationPose weights (Google Drive folder)"
    gdown --folder 1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i -O "$FP_W"
    FP_OK=1
    for run in 2023-10-28-18-33-37 2024-01-11-20-02-45; do
        [ -s "$FP_W/$run/model_best.pth" ] && [ -s "$FP_W/$run/config.yml" ] || { FAILED+=("$FP_W/$run"); FP_OK=0; }
    done
    if [ "$FP_OK" = 0 ]; then
        echo "  NOTE: Google Drive often quota-blocks these files (resets daily)."
        echo "  Retry later, or download the two run folders in a browser from"
        echo "  https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i"
        echo "  and place them under $FP_W/"
    fi
else
    echo "[FAIL] $FP_W -- gdown not installed (pip install gdown)"; FAILED+=("$FP_W")
fi

# Pixel-Perfect Depth + MoGe-2
if [ -d third_party/pixel-perfect-depth ]; then
    gget third_party/pixel-perfect-depth/checkpoints/ppd_moge.pth 1tabmcsbRVDKDfmO4KU1vOjurzN-wp0HV
    get third_party/pixel-perfect-depth/checkpoints/moge2.pt \
        "https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/main/model.pt"
else
    echo "[FAIL] third_party/pixel-perfect-depth missing -- run tools/install.sh first (README, step 1)"
    FAILED+=("third_party/pixel-perfect-depth")
fi

# HOI-DETR hand-object detector
if [ -d third_party/HOI-DETR ]; then
    get third_party/HOI-DETR/checkpoints/epoch_5.pth \
        "https://huggingface.co/ahmaddarkhalil/hoi-detr/resolve/main/epoch_5.pth"
else
    echo "[FAIL] third_party/HOI-DETR missing -- run tools/install.sh first (README, step 1)"
    FAILED+=("third_party/HOI-DETR")
fi

echo
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "DONE WITH ERRORS -- failed/missing:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
echo "All no-login checkpoints in place."
NEEDED=()
{ [ -s data/models/SMPL/SMPL_NEUTRAL.pkl ] && [ -s data/camerahmr/cam_model_cleaned.ckpt ]; } ||
    NEEDED+=("bash tools/fetch_camerahmr_assets.sh        # SMPL + HumanFoV model (camerahmr.is.tue.mpg.de account)")
[ -s data/wilor_data/mano/MANO_RIGHT.pkl ] ||
    NEEDED+=("data/wilor_data/mano/MANO_RIGHT.pkl   # MANO model (mano.is.tue.mpg.de account)")
[ -s data/models/SMPL/SMPLH_NEUTRAL.pkl ] ||
    NEEDED+=("data/models/SMPL/SMPLH_NEUTRAL.pkl    # from the Extended SMPL+H model (mano.is.tue.mpg.de)")
[ -s third_party/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt ] ||
    NEEDED+=("third_party/sam-3d-body/checkpoints/sam-3d-body-dinov3/   # Hugging Face, access request")
[ -s third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml ] ||
    NEEDED+=("third_party/sam-3d-objects/checkpoints/hf/   # Hugging Face, access request")
if [ ${#NEEDED[@]} -gt 0 ]; then
    echo "Still needed (see README, steps 3 and 4):"
    printf '  %s\n' "${NEEDED[@]}"
fi
