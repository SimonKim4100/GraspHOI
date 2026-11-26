"""Person box (ViTDet-H) and intrinsics K (HumanFoV, from CameraHMR)."""
import gc
import os
import sys

import numpy as np
import torch
from detectron2.config import LazyConfig
from torchvision.transforms import Normalize

from human_recon.constants import (
    CAMERAHMR_ROOT, CAM_MODEL_CKPT, DETECTRON_CKPT, DETECTRON_CFG, IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD,
)

if not os.path.isdir(os.path.join(CAMERAHMR_ROOT, 'core')):
    raise ImportError(
        f"CameraHMR not found at {CAMERAHMR_ROOT}. Clone it first:\n"
        f"  git clone https://github.com/pixelite1201/CameraHMR {CAMERAHMR_ROOT}")
# CameraHMR's core package
if CAMERAHMR_ROOT not in sys.path:
    sys.path.insert(0, CAMERAHMR_ROOT)

from core.cam_model.fl_net import FLNet
from core.datasets.utils import resize_image
from core.utils.utils_detectron2 import DefaultPredictor_Lazy


class NoPersonDetected(RuntimeError):
    """The detector found no person."""


def load_detector(threshold=0.25):
    """ViTDet-H person detector with CameraHMR's configuration."""
    cfg = LazyConfig.load(str(DETECTRON_CFG))
    cfg.train.init_checkpoint = DETECTRON_CKPT
    for stage in range(3):                          # the three cascade stages
        cfg.model.roi_heads.box_predictors[stage].test_score_thresh = threshold
    return DefaultPredictor_Lazy(cfg)


def load_fov_model():
    """HumanFoV (FLNet) network, on the CPU."""
    net = FLNet()
    ckpt = torch.load(CAM_MODEL_CKPT, weights_only=False)
    net.load_state_dict(ckpt['state_dict'])
    return net.eval()


def detect_best_person(detector, img_cv2):
    """Highest-scoring person box as (center, scale); raises NoPersonDetected."""
    inst = detector(img_cv2)['instances']
    person = (inst.pred_classes == 0) & (inst.scores > 0.5)    # COCO class 0, confident
    boxes = inst.pred_boxes.tensor[person].cpu().numpy()
    if len(boxes) == 0:
        raise NoPersonDetected("no humans detected in the image")
    if len(boxes) > 1:
        scores = inst.scores[person].cpu().numpy()
        best_idx = np.argmax(scores)
        boxes = boxes[best_idx:best_idx + 1]
    lo, hi = boxes[:, 0:2], boxes[:, 2:4]           # (x1, y1), (x2, y2)
    return (hi + lo) / 2.0, (hi - lo) / 200.0       # center, size / 200


def cam_intrinsics(fov_model, img_cv2):
    """Intrinsic matrix (3, 3) from the FLNet vertical FoV."""
    h, w, _ = img_cv2.shape
    _, square = resize_image(img_cv2, IMAGE_SIZE)
    x = torch.from_numpy(square.astype(np.float32) / 255.0).permute(2, 0, 1)
    x = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)(x)
    with torch.no_grad():
        fov, _ = fov_model(x.unsqueeze(0))
    f = (h / (2 * torch.tan(fov[0, 1] / 2))).item()   # fov[0, 1]: vertical FoV (rad)
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = K[1, 1] = f
    K[0, 2], K[1, 2] = w / 2, h / 2
    return K


def person_box_and_camera(detector, fov_model, img_cv2):
    """(bbox_xyxy, K) of a BGR image; default centered box if no person."""
    H, W = img_cv2.shape[:2]
    K = cam_intrinsics(fov_model, img_cv2)
    try:
        center, scale = detect_best_person(detector, img_cv2)
        wh = scale[0] * 200.0
        x1, y1 = center[0] - wh / 2.0
        x2, y2 = center[0] + wh / 2.0
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
    except NoPersonDetected:
        bbox = np.array([W * 0.15, H * 0.10, W * 0.85, H * 0.95], dtype=np.float32)
        print("[person] warning: no person detected -> default centered box", flush=True)
    return bbox, K.astype(np.float32)


class PersonCamera:
    """Person detector and HumanFoV; call on BGR images, then close()."""

    def __init__(self):
        self.detector = load_detector()
        self.fov_model = load_fov_model()

    def __call__(self, img_cv2):
        """Returns (bbox_xyxy, K)."""
        return person_box_and_camera(self.detector, self.fov_model, img_cv2)

    def close(self):
        self.detector = None
        self.fov_model = None
        gc.collect()
        torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
