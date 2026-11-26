"""Metric scale of the reconstructed mesh."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def _reject_outliers(P, z, mad_k=3.0, knn=16, sor_k=2.0, min_pts=50):
    """Depth then spatial outlier rejection of P (n, 3)."""
    if len(P) < min_pts:
        return P
    # depth outliers (MAD)
    med = np.median(z)
    mad = np.median(np.abs(z - med))
    if mad > 1e-9:
        keep = np.abs(z - med) <= mad_k * 1.4826 * mad
        if keep.sum() >= min_pts:
            P = P[keep]
    if len(P) < min_pts:
        return P
    # statistical outlier removal
    idx = (np.random.RandomState(0).choice(len(P), 20000, replace=False)
           if len(P) > 20000 else np.arange(len(P)))
    nn = NearestNeighbors(n_neighbors=min(knn + 1, len(idx))).fit(P[idx])
    d, _ = nn.kneighbors(P[idx])
    dm = d[:, 1:].mean(1)            # column 0 is the point itself
    thr = dm.mean() + sor_k * dm.std()
    keep = dm <= thr
    if keep.sum() >= min_pts:
        P = P[idx][keep]
    return P


def sphere_rescale(verts, depth, vis_mask, K):
    """Sphere scale from the visible depth cloud. Returns (verts * s, s) or (None, None)."""
    m = (vis_mask > 127) & np.isfinite(depth) & (depth > 1e-6)
    if int(m.sum()) < 50:
        return None, None
    ys, xs = np.nonzero(m)
    z = depth[ys, xs].astype(np.float64)
    fx_, fy_ = float(K[0][0]), float(K[1][1])
    cx, cy = float(K[0][2]), float(K[1][2])
    P = np.stack([(xs - cx) / fx_ * z, (ys - cy) / fy_ * z, z], 1)
    P = _reject_outliers(P, P[:, 2])
    if len(P) < 50:
        return None, None
    r_obs = float(np.linalg.norm(P - np.median(P, 0), axis=1).max())

    V = np.asarray(verts, dtype=np.float64)
    Vc = V - V.mean(0)
    r_mesh = float(np.linalg.norm(Vc - np.median(Vc, 0), axis=1).max())
    if not np.isfinite(r_obs) or not np.isfinite(r_mesh) or r_obs <= 1e-9 or r_mesh <= 1e-9:
        return None, None
    s = r_obs / r_mesh
    return (V * s).astype(np.float32), float(s)


def extent_fallback_rescale(verts, amodal_mask, depth, fx):
    """Extent scale from the amodal bbox at its median depth."""
    verts = np.asarray(verts, dtype=np.float32)
    omask = np.asarray(amodal_mask)
    ys, xs = np.where(omask > 0)
    D_px = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    z_o = float(np.median(depth[(omask > 0) & np.isfinite(depth) & (depth > 1e-6)]))
    target = D_px * z_o / fx
    ext = verts.max(0) - verts.min(0)
    diag = float(np.linalg.norm(ext))
    mscale = target / max(diag, 1e-6)
    # safety cap (m)
    if mscale * diag > 6.0:
        print(f"[scale] extent scale capped (it would give a {mscale*diag:.1f} m diagonal)", flush=True)
        mscale = 6.0 / max(diag, 1e-6)
    c = verts.mean(0)
    return ((verts - c) * mscale + c).astype(np.float32), mscale
