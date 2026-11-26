"""Metric depth from Pixel-Perfect Depth (PPD)."""

import math
import os
import sys

import numpy as np
import torch

_PPD_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "third_party", "pixel-perfect-depth")
_PPD_CKPT = os.path.join(_PPD_ROOT, "checkpoints", "ppd_moge.pth")
_MOGE_CKPT = os.path.join(_PPD_ROOT, "checkpoints", "moge2.pt")

PPD_STEPS = 20              # diffusion sampling steps
_SEED = 666                 # per-call seed
_MAX_FIT_PX = 100_000       # max. pixels for the RANSAC fit
SATURATED_FRAC = 0.3        # saturated above this fraction at max depth

_MODEL = None


def fov_x_from_intrinsics(K, image_width):
    """Horizontal field of view (deg) of K."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx = float(K[0, 0])
    return math.degrees(2.0 * math.atan(image_width / (2.0 * fx)))


def _load_rgb(image):
    """RGB copy of an HxWx3 cv2 BGR frame."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 image, got {arr.shape}")
    return arr[..., :3][..., ::-1].copy()


def _get_model():
    global _MODEL
    if _MODEL is None:
        if _PPD_ROOT not in sys.path:
            sys.path.insert(0, _PPD_ROOT)
        from ppd.models.ppd import PixelPerfectDepth

        model = PixelPerfectDepth(semantics_model="MoGe2", semantics_pth=_MOGE_CKPT,
                                  sampling_steps=PPD_STEPS)
        model.load_state_dict(torch.load(_PPD_CKPT, map_location="cpu"), strict=False)
        _MODEL = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    elif torch.cuda.is_available():
        # undo park_cpu()
        try:
            if next(_MODEL.parameters()).device.type == "cpu":
                _MODEL.to("cuda")
        except StopIteration:
            pass
    return _MODEL


def park_cpu():
    """Park the PPD model on the CPU."""
    global _MODEL
    if _MODEL is not None and torch.cuda.is_available():
        try:
            if next(_MODEL.parameters()).device.type == "cuda":
                _MODEL.to("cpu")
                torch.cuda.empty_cache()
        except StopIteration:
            pass


def predict_depth(image, K):
    """HxW metric depth (m) of a cv2 BGR frame."""
    model = _get_model()
    dev = next(model.dit.parameters()).device

    rgb = _load_rgb(image)
    h, w = rgb.shape[:2]
    fov_x = fov_x_from_intrinsics(K, w)

    img_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).to(dev)
    with torch.no_grad():
        # depth is inf outside the mask
        moge_depth, valid, _ = model.sem_encoder.infer(img_t, fov_x=fov_x)
    moge_depth = np.asarray(moge_depth, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(moge_depth) & (moge_depth > 1e-6)

    torch.manual_seed(_SEED)
    bgr = np.ascontiguousarray(rgb[..., ::-1])  # infer_image expects BGR
    with torch.no_grad():
        rel, _ = model.infer_image(bgr)
    rel = torch.nn.functional.interpolate(rel.float(), size=(h, w), mode="bilinear",
                                          align_corners=False)[0, 0].cpu().numpy()

    from ppd.utils.align_depth_func import recover_metric_depth_ransac

    fit = valid
    if int(fit.sum()) > _MAX_FIT_PX:
        keep = np.zeros(int(fit.sum()), bool)
        keep[np.random.default_rng(0).choice(int(fit.sum()), _MAX_FIT_PX, replace=False)] = True
        fit = fit.copy()
        fit[fit] = keep
    np.random.seed(_SEED)  # seeds sklearn's RANSAC
    metric = recover_metric_depth_ransac(rel, moge_depth, fit).astype(np.float32)
    # saturated alignment: use MoGe-2 depth
    if valid.any():
        sat = float(np.mean(metric[valid] >= metric.max() * (1 - 1e-6)))
        if sat > SATURATED_FRAC:
            print(f"[depth] PPD alignment saturated ({100 * sat:.0f}% of valid pixels at the maximum) "
                  f"-> MoGe-2 depth", flush=True)
            metric = np.where(valid, moge_depth, moge_depth[valid].max()).astype(np.float32)
    return metric


def predict_depth_amodal(amodal_rgb, scene_depth, amodal_mask, visible_mask):
    """PPD depth of the completed object, aligned and pasted into scene_depth."""
    model = _get_model()
    h, w = scene_depth.shape[:2]

    amask = np.asarray(amodal_mask) > 0
    fit = amask & np.isfinite(scene_depth) & (scene_depth > 1e-6)
    fit &= np.asarray(visible_mask) > 0
    if int(fit.sum()) < 100:
        return scene_depth

    torch.manual_seed(_SEED)
    bgr = np.ascontiguousarray(np.asarray(amodal_rgb)[..., :3][..., ::-1])
    with torch.no_grad():
        rel, _ = model.infer_image(bgr)
    rel = torch.nn.functional.interpolate(rel.float(), size=(h, w), mode="bilinear",
                                          align_corners=False)[0, 0].cpu().numpy()

    from ppd.utils.align_depth_func import recover_metric_depth_ransac

    if int(fit.sum()) > _MAX_FIT_PX:
        keep = np.zeros(int(fit.sum()), bool)
        keep[np.random.default_rng(0).choice(int(fit.sum()), _MAX_FIT_PX, replace=False)] = True
        fit = fit.copy()
        fit[fit] = keep
    np.random.seed(_SEED)
    aligned = recover_metric_depth_ransac(rel, scene_depth, fit).astype(np.float32)

    out = scene_depth.copy()
    out[amask] = aligned[amask]
    return out
