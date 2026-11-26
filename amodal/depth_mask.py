"""Depth-based occluder region for amodal completion."""
import numpy as np


def behind_mask(visible_mask, depth, *, far_pct=90):
    """Pixels deeper than the far_pct percentile of the visible object."""
    vm = np.asarray(visible_mask).astype(bool)
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth) & (depth > 1e-6)
    vd = depth[vm & finite]
    if vd.size == 0:
        return np.zeros(vm.shape, dtype=bool)
    z_far = float(np.percentile(vd, far_pct))
    return finite & (depth > z_far)


def depth_occluder_mask(visible_mask, depth, *, far_pct=90, margin_frac=0.25,
                        margin_px=(8, 60)):
    """Initial occluder region (HxW uint8) inside the expanded object box."""
    vm = np.asarray(visible_mask).astype(bool)
    H, W = vm.shape
    depth = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(depth) & (depth > 1e-6)

    ys, xs = np.where(vm)
    diag = float(np.hypot(np.ptp(ys) + 1, np.ptp(xs) + 1))
    margin = int(np.clip(round(margin_frac * diag), margin_px[0], margin_px[1]))
    by0 = max(0, ys.min() - margin); by1 = min(H, ys.max() + 1 + margin)
    bx0 = max(0, xs.min() - margin); bx1 = min(W, xs.max() + 1 + margin)
    box = np.zeros((H, W), bool); box[by0:by1, bx0:bx1] = True
    candidate = box & ~vm

    # drop background and invalid depth
    behind = behind_mask(vm, depth, far_pct=far_pct)
    return (candidate & ~behind & finite).astype(np.uint8)
