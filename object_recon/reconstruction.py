"""SAM 3D Objects reconstruction and mesh cleaning."""
from collections import defaultdict

import fast_simplification
import numpy as np
import trimesh
from scipy.spatial import cKDTree

TARGET_FACES = 40_000    # decimation target


def amodal_input(img_rgb, vis_mask, amodal_on_white, amodal_sil):
    """Photo with the amodal completion pasted in, and the amodal mask."""
    sv = np.asarray(vis_mask) > 127
    sa = (np.asarray(amodal_sil) > 127) | sv
    comp = np.ascontiguousarray(img_rgb[..., :3]).copy()
    occ = sa & ~sv
    comp[occ] = np.asarray(amodal_on_white)[..., :3][occ]
    return comp.astype(np.uint8), sa


def _fib_dirs(n):
    """n near-uniform unit directions (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def visible_faces(m, n_dirs=300, res=256):
    """Bool mask of the faces seen from n_dirs directions."""
    c = m.bounds.mean(0); R = np.linalg.norm(m.extents) * 0.55
    g = (np.arange(res) + 0.5) / res * 2 * R - R
    A, B = [x.reshape(-1, 1) for x in np.meshgrid(g, g)]
    vis = np.zeros(len(m.faces), bool)
    for d in _fib_dirs(n_dirs):
        a = np.cross(d, [0, 0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0, 0]); a /= np.linalg.norm(a)
        b = np.cross(d, a)
        hit = m.ray.intersects_first(c - d * 2 * R + A * a + B * b, np.repeat(d[None], res * res, 0))
        vis[hit[hit >= 0]] = True
    return vis


def clean_mesh(V, F, rgb, target_faces=TARGET_FACES, min_frac=0.005):
    """Drop inverted shells, specks and hidden faces, and decimate."""
    if not trimesh.ray.has_embree:
        raise RuntimeError("Mesh cleaning needs trimesh's embree ray backend: pip install embreex")
    tm = trimesh.Trimesh(np.asarray(V, np.float64), np.asarray(F, np.int64), process=False)
    comps = tm.split(only_watertight=False)
    nf = sum(len(c.faces) for c in comps)
    keep, n_inv = [], 0
    for c in comps:
        if c.is_watertight and c.volume < 0:
            n_inv += 1
            continue
        if len(c.faces) < min_frac * nf:
            continue
        if not c.is_watertight and len(c.faces) < 0.02 * nf:
            continue
        keep.append(c)
    if not keep:
        keep = [max(comps, key=lambda c: len(c.faces))]
    m = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    if len(m.faces) > target_faces:
        v2, f2 = fast_simplification.simplify(np.asarray(m.vertices, np.float32),
                                              np.asarray(m.faces, np.int32),
                                              target_reduction=1.0 - target_faces / len(m.faces))
        m = trimesh.Trimesh(v2, f2, process=False)
    m.remove_unreferenced_vertices()
    if m.is_watertight and m.volume < 0:
        m.invert()
    n_hidden = 0
    vis = visible_faces(m)
    if vis.any():
        n_hidden = int((~vis).sum())
        m = trimesh.Trimesh(m.vertices, m.faces[vis], process=False)
        m.remove_unreferenced_vertices()
    _, nn = cKDTree(np.asarray(V, np.float64)).query(np.asarray(m.vertices))
    col = np.asarray(rgb)[nn]
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.concatenate(
        [col[:, :3], np.full((len(col), 1), 255, np.uint8)], 1))
    stats = dict(n_comp=len(comps), n_inverted=n_inv, n_kept=len(keep), faces_in=int(nf),
                 faces_hidden=n_hidden, faces_out=len(m.faces), watertight=bool(m.is_watertight))
    return m, stats


def close_holes(V, F, tol=1e-7):
    """Weld vertices, cap holes and orient outward. Returns the new faces."""
    V = np.asarray(V, np.float64)
    _, first, inv = np.unique(np.round(V / tol).astype(np.int64), axis=0, return_index=True,
                              return_inverse=True)
    Fw = first[inv.ravel()][np.asarray(F, np.int64)]
    he = set(map(tuple, np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]).tolist()))
    out = defaultdict(list)
    for a, b in he:
        if (b, a) not in he:
            out[a].append(b)
    caps = []
    for s in list(out):
        while out[s]:
            cyc, cur, guard = [s], out[s].pop(), 0
            while cur != s and out[cur] and guard < 1_000_000:
                cyc.append(cur)
                cur = out[cur].pop()
                guard += 1
            if len(cyc) >= 3:                      # fan cap
                h = int(np.argmin(((V[cyc] - V[cyc].mean(0)) ** 2).sum(1)))
                o = [cyc[(h + i) % len(cyc)] for i in range(len(cyc))]
                caps.extend((o[0], o[i + 1], o[i]) for i in range(1, len(o) - 1))
    if caps:
        Fw = np.vstack([Fw, np.asarray(caps, np.int64)])
    Fw = Fw[(Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])]
    _, u = np.unique(np.sort(Fw, axis=1), axis=0, return_index=True)
    Fw = Fw[np.sort(u)]
    if trimesh.Trimesh(V, Fw, process=False).volume < 0:
        Fw = Fw[:, ::-1]
    return np.ascontiguousarray(Fw)


def reconstruct_object(worker, img_rgb, vis_mask, amodal_on_white, amodal_sil):
    """Reconstruct one object. Returns (cleaned mesh, worker reply with the layout)."""
    image, mask = amodal_input(img_rgb, vis_mask, amodal_on_white, amodal_sil)
    res = worker.reconstruct(image, mask)
    mesh, stats = clean_mesh(res['V_canon'], res['F'], res['rgb'])
    if not mesh.is_watertight:
        mesh = trimesh.Trimesh(mesh.vertices, close_holes(mesh.vertices, mesh.faces),
                               vertex_colors=mesh.visual.vertex_colors, process=False)
    print(f"[recon] SAM 3D Objects mesh: {stats['faces_in']} -> {len(mesh.faces)} faces "
          f"({stats['n_kept']}/{stats['n_comp']} components kept, {stats['n_inverted']} inverted, "
          f"{stats['faces_hidden']} hidden faces removed, "
          f"{'holes capped' if not stats['watertight'] else 'watertight'})", flush=True)
    return mesh, res
