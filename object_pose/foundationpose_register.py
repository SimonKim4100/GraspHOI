"""FoundationPose registration of the object mesh."""
import logging
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


RUNTIME_DIR = Path(__file__).resolve().parent / "foundationpose_runtime"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

_FOUNDATIONPOSE_CLASS = None
_SCORE_PREDICTOR_CLASS = None
_REFINE_PREDICTOR_CLASS = None
_SCORER = None
_REFINER = None


def register_object_pose(*, mask, mesh, bgr, K, depth, iteration=5):
    """4x4 object-to-camera pose of a metric mesh in the masked RGB-D frame."""
    if not torch.cuda.is_available():
        raise RuntimeError("FoundationPose register() requires CUDA.")

    _check_foundationpose_assets()

    rgb_np = _prepare_rgb(bgr)
    K_np = np.asarray(K, dtype=np.float32).reshape(3, 3)
    mask = _prepare_mask(mask, rgb_np.shape[:2])
    depth = _prepare_depth(depth, rgb_np.shape[:2])

    foundationpose_cls = _get_foundationpose_class()
    scorer, refiner = _get_predictors()
    estimator = foundationpose_cls(
        model_pts=mesh.vertices.copy(),
        model_normals=mesh.vertex_normals.copy(),
        symmetry_tfs=None,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        glctx=None,
        debug=0,
        debug_dir=tempfile.gettempdir(),   # unused at debug=0
    )
    return estimator.register(K=K_np, rgb=rgb_np, depth=depth, ob_mask=mask, iteration=iteration)


def _get_foundationpose_class():
    global _FOUNDATIONPOSE_CLASS
    if _FOUNDATIONPOSE_CLASS is None:
        # its Utils reloads logging at INFO; keep the caller's logging.disable level
        level = logging.root.manager.disable
        from estimater import FoundationPose
        logging.disable(level)

        _FOUNDATIONPOSE_CLASS = FoundationPose
    return _FOUNDATIONPOSE_CLASS


def _get_predictors():
    global _SCORER, _REFINER, _SCORE_PREDICTOR_CLASS, _REFINE_PREDICTOR_CLASS
    if _SCORE_PREDICTOR_CLASS is None:
        from learning.training.predict_score import ScorePredictor

        _SCORE_PREDICTOR_CLASS = ScorePredictor
    if _REFINE_PREDICTOR_CLASS is None:
        from learning.training.predict_pose_refine import PoseRefinePredictor

        _REFINE_PREDICTOR_CLASS = PoseRefinePredictor
    if _SCORER is None:
        _SCORER = _SCORE_PREDICTOR_CLASS()
    if _REFINER is None:
        _REFINER = _REFINE_PREDICTOR_CLASS()
    return _SCORER, _REFINER


def _check_foundationpose_assets():
    missing = []
    # scorer and refiner run directories
    for run_name in ("2024-01-11-20-02-45", "2023-10-28-18-33-37"):
        run_dir = RUNTIME_DIR / "weights" / run_name
        for filename in ("config.yml", "model_best.pth"):
            path = run_dir / filename
            if not path.exists():
                missing.append(str(path))
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            "FoundationPose runtime weights are missing. Expected:\n"
            f"  {joined}\n"
            "Place the pretrained FoundationPose scorer/refiner weights under "
            f"{RUNTIME_DIR / 'weights'}."
        )


def _prepare_rgb(bgr):
    image = np.asarray(bgr)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected RGB/BGR image with shape HxWx3, got {image.shape}")
    return image[..., :3][..., ::-1].copy()


def _prepare_mask(mask, image_shape):
    mask = np.asarray(mask)
    h, w = image_shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def _prepare_depth(depth, image_shape):
    depth_np = np.asarray(depth).astype(np.float32)
    h, w = image_shape
    if depth_np.shape[:2] != (h, w):
        depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth_np
