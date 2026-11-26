"""Paths and constants of the human stage and workers."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA = REPO_ROOT / 'data'
_THIRD_PARTY = REPO_ROOT / 'third_party'

# HumanFoV and ViTDet-H person detector
CAMERAHMR_ROOT = str(_THIRD_PARTY / 'CameraHMR')
CAM_MODEL_CKPT = str(_DATA / 'camerahmr' / 'cam_model_cleaned.ckpt')
DETECTRON_CKPT = str(_DATA / 'camerahmr' / 'model_final_f05665.pkl')
DETECTRON_CFG = str(_THIRD_PARTY / 'CameraHMR' / 'core' / 'utils' / 'cascade_mask_rcnn_vitdet_h_75ep.py')

SMPL_MODEL_PATH = str(_DATA / 'models' / 'SMPL' / 'SMPL_NEUTRAL.pkl')
SMPLH_MODEL_PATH = str(_DATA / 'models' / 'SMPL' / 'SMPLH_NEUTRAL.pkl')
NUM_BETAS = 10

# HumanFoV input
IMAGE_SIZE = 256
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# HOI-DETR worker
HOIDETR_ROOT = str(_THIRD_PARTY / 'HOI-DETR')
HOIDETR_CONFIG = str(_THIRD_PARTY / 'HOI-DETR' / 'projects' / 'configs' / 'co_dino_vit' /
                     'co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py')
HOIDETR_CKPT = str(_THIRD_PARTY / 'HOI-DETR' / 'checkpoints' / 'epoch_5.pth')

# SAM 3D Body worker
SAM3D_BODY_ROOT = str(_THIRD_PARTY / 'sam-3d-body')
SAM3D_BODY_CKPT = str(_THIRD_PARTY / 'sam-3d-body' / 'checkpoints' / 'sam-3d-body-dinov3' / 'model.ckpt')
SAM3D_BODY_MHR_MODEL = str(_THIRD_PARTY / 'sam-3d-body' / 'checkpoints' / 'sam-3d-body-dinov3' / 'assets' /
                           'mhr_model.pt')
MHR_FITTER = str(_THIRD_PARTY / 'meta_mhr_conversion' / 'pytorch_fitting.py')
MHR_MAPPING = str(_THIRD_PARTY / 'meta_mhr_conversion' / 'assets' / 'mhr2smpl_mapping.npz')

# SAM 3D Objects worker
SAM3D_OBJECTS_ROOT = str(_THIRD_PARTY / 'sam-3d-objects')
