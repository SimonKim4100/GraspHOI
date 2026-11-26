"""Placement from the SAM 3D Objects layout."""
import cv2
import numpy as np
import torch

LAYOUT_IOU_GATE = 0.7                   # min. IoU after the re-fit
_FLIP = np.array([-1.0, -1.0, 1.0])     # PyTorch3D <-> OpenCV camera axes


def as_numpy(x, dtype=np.float64):
    """Tensor, list or array -> numpy array of dtype."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def quat_wxyz_to_matrix(q):
    """Rotation matrix of a (w, x, y, z) quaternion (numpy port from pytorch3d)."""
    w, x, y, z = np.asarray(q, np.float64).reshape(4)
    two_s = 2.0 / (w * w + x * x + y * y + z * z)
    return np.array([
        [1 - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w)],
        [two_s * (x * y + z * w), 1 - two_s * (x * x + z * z), two_s * (y * z - x * w)],
        [two_s * (x * z - y * w), two_s * (y * z + x * w), 1 - two_s * (x * x + y * y)]])


def pose_sam_camera(V_canon, res):
    """Canonical vertices -> SAM 3D's OpenCV camera, in its units."""
    R = quat_wxyz_to_matrix(res['q_wxyz'])
    return ((np.asarray(V_canon, np.float64) * res['s']) @ R + res['t_p3d']) * _FLIP


def _rot_between(a, b):
    """Rotation taking the direction of a onto the direction of b."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b); s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def to_pipeline_camera(V_s, K_s, K_o):
    """Re-express V_s from SAM 3D's camera in the pipeline camera."""
    V_s = np.asarray(V_s, np.float64)
    c = 0.5 * (V_s.min(0) + V_s.max(0))
    u = np.array([K_s[0, 0] * c[0] / c[2] + K_s[0, 2], K_s[1, 1] * c[1] / c[2] + K_s[1, 2]])
    d_new = np.array([(u[0] - K_o[0, 2]) / K_o[0, 0], (u[1] - K_o[1, 2]) / K_o[1, 1], 1.0])
    d_new /= np.linalg.norm(d_new)
    r_s = np.linalg.norm(c)
    r_new = r_s * np.sqrt(K_o[0, 0] * K_o[1, 1] / (K_s[0, 0] * K_s[1, 1])) * (c[2] / r_s / d_new[2]) ** 1.5
    R = _rot_between(c, d_new)
    return (V_s - c) @ R.T + r_new * d_new


def silhouette(V, F, K, hw):
    """Union of the projected triangles, (H, W) bool."""
    H, W = hw
    V = np.asarray(V, np.float64); F = np.asarray(F)
    F = F[(V[F, 2] > 1e-6).all(1)]
    m = np.zeros((H, W), np.uint8)
    if len(F) == 0:
        return m.astype(bool)
    uv = V[:, :2] / V[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    tri = uv[F]
    lo, hi = tri.min(1), tri.max(1)
    tri = tri[(hi[:, 0] >= -1) & (lo[:, 0] <= W) & (hi[:, 1] >= -1) & (lo[:, 1] <= H)]
    for t3 in np.round(tri * 16).astype(np.int32):          # 1/16-px fixed point
        cv2.fillConvexPoly(m, t3, 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def iou(a, b):
    """IoU of two bool masks."""
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def amodal_target(amodal_sil, vis_mask):
    """Placement target as a bool mask: amodal silhouette | visible mask."""
    sv = np.asarray(vis_mask) > 127
    return (np.asarray(amodal_sil) > 127) | sv


def fit_iou(V, F, cam_int, amodal_sil, vis_mask):
    """Silhouette IoU of camera-frame vertices against the target."""
    sa = amodal_target(amodal_sil, vis_mask)
    return iou(silhouette(as_numpy(V).reshape(-1, 3), as_numpy(F, np.int64).reshape(-1, 3),
                          as_numpy(cam_int).reshape(3, 3), sa.shape), sa)


def zbuffer(V, F, K, hw, device='cpu'):
    """Front-surface camera depth per pixel, (H, W), 0 where empty."""
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection
    H, W = hw
    dev = torch.device(device)
    cams = cameras_from_opencv_projection(
        R=torch.eye(3, device=dev)[None], tvec=torch.zeros(1, 3, device=dev),
        camera_matrix=torch.as_tensor(K, dtype=torch.float32, device=dev)[None],
        image_size=torch.tensor([[H, W]], device=dev))
    mesh = Meshes(verts=[torch.as_tensor(np.asarray(V), dtype=torch.float32, device=dev)],
                  faces=[torch.as_tensor(np.asarray(F), dtype=torch.int64, device=dev)])
    rs = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0)
    frag = MeshRasterizer(cameras=cams, raster_settings=rs)(mesh)
    z = frag.zbuf[0, ..., 0].cpu().numpy()
    return np.where(z > 0, z, 0.0)


def depth_pull(V, F, K, depth, vis_mask, amodal_mask, device='cpu', erode_px=2, min_px=20):
    """Scale putting V's front surface on the depth map. Returns (scale or None, n_px, src)."""
    H, W = depth.shape
    z = zbuffer(V, F, K, (H, W), device)
    ok = (z > 0) & np.isfinite(depth) & (depth > 1e-6)
    sv = (np.asarray(vis_mask) > 127).astype(np.uint8)
    cands = [('visible', cv2.erode(sv, np.ones((2 * erode_px + 1,) * 2, np.uint8)) > 0),
             ('visible_uneroded', sv > 0),
             ('amodal', np.asarray(amodal_mask, bool))]
    n = 0
    for src, m in cands:
        sel = ok & m
        n = int(sel.sum())
        if n >= min_px:
            return float(np.median(depth[sel] / z[sel])), n, src
    return None, n, None


def refit_translation(V, F, K, hw, target, iters=50, pad_frac=0.5, ss=4):
    """Translate V to match target's centroid and area. Returns (V, IoU, info)."""
    H, W = hw
    target = np.asarray(target, bool)
    V = np.asarray(V, np.float64)
    ty, tx = np.nonzero(target)
    if not len(tx):
        return V, 0.0, dict(n_iter=0, depth_factor=1.0, shift_px=0.0, moments=None)
    ct, At = np.array([tx.mean(), ty.mean()]), float(len(tx))
    cut = target[0].any() or target[-1].any() or target[:, 0].any() or target[:, -1].any()
    p = int(round(pad_frac * max(H, W)))
    Kp = np.array(K, np.float64)
    Kp[0, 2] += p
    Kp[1, 2] += p

    def centre(X):
        return 0.5 * (X.min(0) + X.max(0))

    def pixel(c):
        return np.array([K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]])

    def moments(X, coarse, lo):
        """Supersampled area and centroid of X's silhouette."""
        ys, xs = np.nonzero(coarse)
        x0, y0 = xs.min() + lo, ys.min() + lo
        w, h = xs.max() + lo + 1 - x0, ys.max() + lo + 1 - y0
        s_ = int(np.clip(2048 // max(w, h), 1, ss))
        Kh = np.array([[s_ * K[0, 0], 0.0, s_ * (K[0, 2] - x0) + (s_ - 1) / 2],
                       [0.0, s_ * K[1, 1], s_ * (K[1, 2] - y0) + (s_ - 1) / 2], [0.0, 0.0, 1.0]])
        fy, fx = np.nonzero(silhouette(X, F, Kh, (s_ * h, s_ * w)))
        if not len(fx):
            return 0.0, None
        return len(fx) / s_ ** 2, np.array([(fx + 0.5) / s_ - 0.5 + x0, (fy + 0.5) / s_ - 0.5 + y0]).mean(1)

    first = silhouette(V, F, Kp, (H + 2 * p, W + 2 * p))
    s0 = iou(first[p:p + H, p:p + W], target)
    best = (s0, V, 0, None)
    for mode in (('full', 'frame') if cut else ('full',)):
        cur, full, s_cur, n = V, first, s0, 0
        for it in range(1, iters + 1):
            coarse, lo = (full[p:p + H, p:p + W], 0) if mode == 'frame' else (full, -p)
            c = centre(cur)
            if not coarse.any() or not c[2] > 1e-9:
                break
            A, cs = moments(cur, coarse, lo)
            if cs is None:
                break
            k = float(np.sqrt(A / At))
            u = ct - (cs - pixel(c)) / k
            z = c[2] * k
            cand = cur - c + np.array([(u[0] - K[0, 2]) / K[0, 0] * z, (u[1] - K[1, 2]) / K[1, 1] * z, z])
            cfull = silhouette(cand, F, Kp, (H + 2 * p, W + 2 * p))
            s = iou(cfull[p:p + H, p:p + W], target)
            if not s > s_cur:
                break
            cur, full, s_cur, n = cand, cfull, s, it
        if s_cur > best[0]:
            best = (s_cur, cur, n, mode)
    s, Vb, it, mode = best
    c0, c1 = centre(V), centre(Vb)
    return Vb, s, dict(n_iter=it, depth_factor=float(np.linalg.norm(c1) / np.linalg.norm(c0)),
                       shift_px=float(np.linalg.norm(pixel(c1) - pixel(c0))), moments=mode)


def refit_and_pull(V, F, cam_int, amodal_sil, vis_mask, depth_map, device='cuda'):
    """Re-fit V to the target and pull it onto the depth map. Returns (V, info)."""
    K = as_numpy(cam_int).reshape(3, 3)
    depth_map = as_numpy(depth_map, np.float32)
    sa = amodal_target(amodal_sil, vis_mask)
    F = np.asarray(F)
    V = np.asarray(V, np.float64)
    info = {}
    V, info['iou'], info['refit'] = refit_translation(V, F, K, depth_map.shape, sa)
    N, info['n_depth_px'], info['pull_src'] = depth_pull(V, F, K, depth_map, vis_mask, sa, device)
    info['N'] = N
    return (V * N if N is not None else V).astype(np.float32), info


def place_from_layout(mesh_tm, res, cam_int, amodal_sil, vis_mask, depth_map, device='cuda'):
    """Place the SAM 3D mesh. Returns a dict, without verts if too few depth pixels."""
    K_o = as_numpy(cam_int).reshape(3, 3)
    depth_map = as_numpy(depth_map, np.float32)
    H, W = depth_map.shape
    V_s = pose_sam_camera(np.asarray(mesh_tm.vertices), res)
    V_o = to_pipeline_camera(V_s, np.asarray(res['K'], np.float64), K_o)
    sa = amodal_target(amodal_sil, vis_mask)
    F = np.asarray(mesh_tm.faces)
    out = dict(iou_raw=iou(silhouette(V_o, F, K_o, (H, W)), sa))
    V_o, out['iou'], out['refit'] = refit_translation(V_o, F, K_o, (H, W), sa)
    out['use_layout'] = out['iou'] >= LAYOUT_IOU_GATE
    if not out['use_layout']:
        out['why'] = f'iou < {LAYOUT_IOU_GATE}'
    N, n_px, src = depth_pull(V_o, F, K_o, depth_map, vis_mask, sa, device)
    if N is None:                                      # too few depth pixels
        if out['use_layout']:
            out['why'] = f'depth pull: {n_px} px'
        out['use_layout'] = False
        return out
    out.update(verts=(V_o * N).astype(np.float32), N=N, pull_src=src, n_depth_px=n_px)
    return out
