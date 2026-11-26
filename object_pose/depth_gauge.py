"""Depth gauges: median ratio of vertex depth to the depth map."""
import numpy as np
import trimesh

from object_pose.layout import as_numpy
from optimizer.common import LEFT_HAND_FULL, RIGHT_HAND_FULL

MIN_HAND_VERTS = 20     # min. vertices for the hand gauge


def body_depth_gauge(smpl_verts, depth, K):
    """Body gauge from camera-frame vertices; NaN if none has a valid depth."""
    fx, fy = float(K[0, 0]), float(K[1, 1]); H, W = depth.shape
    ratios = []
    for vp in smpl_verts:
        if vp[2] <= 1e-6:
            continue
        u = int(round(fx * vp[0] / vp[2] + W / 2.0)); v = int(round(fy * vp[1] / vp[2] + H / 2.0))
        if 0 <= u < W and 0 <= v < H and np.isfinite(depth[v, u]) and depth[v, u] > 1e-6:
            ratios.append(vp[2] / depth[v, u])
    return float(np.median(ratios))


def hand_depth_gauge(depth, smplh_verts, smplh_faces, K, vis_mask, left_flag, right_flag,
                     min_pts=MIN_HAND_VERTS):
    """Hand gauge of the detected hands. Returns (gauge, n_verts); 1.0 if too few."""
    depth = as_numpy(depth, np.float32)
    K = as_numpy(K).reshape(3, 3)
    idx = [np.asarray(h) for h, f in ((LEFT_HAND_FULL, left_flag), (RIGHT_HAND_FULL, right_flag)) if f]
    if not idx:
        return 1.0, 0
    sv = as_numpy(smplh_verts).reshape(-1, 3)
    nrm = trimesh.Trimesh(sv, as_numpy(smplh_faces, np.int64).reshape(-1, 3), process=False).vertex_normals
    idx = np.concatenate(idx)
    p = sv[idx]
    facing = (np.einsum('ij,ij->i', nrm[idx], p) < 0) & (p[:, 2] > 1e-6)
    H, W = depth.shape
    u = np.round(K[0, 0] * p[:, 0] / np.maximum(p[:, 2], 1e-6) + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * p[:, 1] / np.maximum(p[:, 2], 1e-6) + K[1, 2]).astype(int)
    ok = facing & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ok[ok] &= ~(np.asarray(vis_mask)[v[ok], u[ok]] > 127)
    d = depth[v[ok], u[ok]]
    good = np.isfinite(d) & (d > 1e-6)
    if good.sum() < min_pts:
        return 1.0, int(good.sum())
    return float(np.median(p[ok][good, 2] / d[good])), int(good.sum())
