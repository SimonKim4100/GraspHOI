"""Contact decision and object seating."""
import cv2
import igl
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.ops import knn_points
from scipy.spatial import cKDTree

from find_contact.hoi_detr_contact import associate, bbox_from_verts_2d
from optimizer.common import (LEFT_FINGERTIPS, LEFT_HAND_FULL, LEFT_HAND_VERTS, RIGHT_FINGERTIPS,
                              RIGHT_HAND_FULL, RIGHT_HAND_VERTS, project_to_image)
from optimizer.objoptim import ObjectOptimizer, object_depth_N

MATCH_PX = 6.0      # same-ray match radius (px)
DEMOTE_PX = 25      # min. hand box overlap with the object (px)
PEN_THR = 0.002     # penetration limit (m)
PEN_W = 20.0        # penetration weight
WHITE_THR = 245     # background above this in all channels


def contact_region(amodal_on_white):
    """Non-white mask of the amodal cutout."""
    im = np.asarray(amodal_on_white)[..., :3]
    return ~((im[..., 0] > WHITE_THR) & (im[..., 1] > WHITE_THR) & (im[..., 2] > WHITE_THR))


def detect_and_seat(optimizer_model, obj_verts_fit, depth_map, record, amodal, vis_mask, smplh_verts,
                    smplh_faces, obj_faces, focal_length, img_w, img_h, left_flag, right_flag):
    """Decide per-hand contact and seat the object on the contacting hands."""
    # depth-map scale of an object fit
    N_depth = (object_depth_N(optimizer_model, obj_verts_fit, depth_map)
               if isinstance(optimizer_model, ObjectOptimizer) else 1.0)
    # floats keep the projection on the CPU
    body_verts = np.asarray(smplh_verts[0])
    fx, fy = float(focal_length[0]), float(focal_length[1])
    iw, ih = float(img_w), float(img_h)
    cam_bb = {}
    for side, full_idx, flag in (("left", LEFT_HAND_FULL, left_flag),
                                 ("right", RIGHT_HAND_FULL, right_flag)):
        if flag:
            proj = project_to_image(
                torch.as_tensor(body_verts[np.asarray(full_idx)], dtype=torch.float32), (fx, fy), iw, ih)
            cam_bb[side] = bbox_from_verts_2d(proj.detach().cpu().numpy())
        else:
            cam_bb[side] = None
    # object box in record pixels
    obj_box = None
    amodal_np = np.asarray(amodal); amodal_h, amodal_w = amodal_np.shape[:2]
    amodal_ys, amodal_xs = np.nonzero(amodal_np > 0)
    if len(amodal_ys):
        rec_sx, rec_sy = float(record['width']) / amodal_w, float(record['height']) / amodal_h
        obj_box = [float(amodal_xs.min()) * rec_sx, float(amodal_ys.min()) * rec_sy,
                   float(amodal_xs.max()) * rec_sx, float(amodal_ys.max()) * rec_sy]
    det = associate(record, cam_bb, img_wh=(float(img_w), float(img_h)), obj_bbox=obj_box)
    left_contact, right_contact = det['left'][0], det['right'][0]
    # demote hands barely overlapping the object
    if left_contact or right_contact:
        amodal_px = amodal_np > 0
        mask_sx, mask_sy = amodal_w / max(float(img_w), 1e-6), amodal_h / max(float(img_h), 1e-6)
        for side in ("left", "right"):
            in_contact = left_contact if side == "left" else right_contact
            if not in_contact or cam_bb[side] is None:
                continue
            box = cam_bb[side]
            x0 = max(int(box[0] * mask_sx), 0); x1 = min(int(box[2] * mask_sx) + 1, amodal_w)
            y0 = max(int(box[1] * mask_sy), 0); y1 = min(int(box[3] * mask_sy) + 1, amodal_h)
            overlap_px = int(amodal_px[y0:y1, x0:x1].sum()) if (x1 > x0 and y1 > y0) else 0
            if overlap_px < DEMOTE_PX:
                print(f"[contact] {side} hand demoted: its box overlaps {overlap_px}px of the "
                      f"amodal object", flush=True)
                det[side] = (False,) + tuple(det[side][1:])
                if side == "left":
                    left_contact = False
                else:
                    right_contact = False
    print(f"[contact] left={left_contact} right={right_contact}", flush=True)

    contacting = [n for n in ('left', 'right') if det[n][0]]
    contact_pairs = None
    if len(contacting) >= 1:
        with torch.no_grad():
            obj = optimizer_model().detach()
        dev = obj.device

        hand_idx = np.concatenate([np.asarray(RIGHT_HAND_FULL if n == "right" else LEFT_HAND_FULL)
                                   for n in contacting]).astype(int)
        hand_verts = torch.as_tensor(np.asarray(smplh_verts[0])[hand_idx], dtype=torch.float32,
                                     device=dev)
        obj_sub = obj[::max(1, len(obj) // 4000)]
        hand_px = project_to_image(hand_verts, focal_length, img_w, img_h)
        hand_px_int = hand_px.round().long()
        amodal_arr = np.asarray(amodal)
        mask_h, mask_w = amodal_arr.shape[:2]
        hand_px_np = hand_px_int.detach().cpu().numpy()
        body_faces = np.asarray(smplh_faces)
        hand_faces = body_faces[np.isin(body_faces, hand_idx).all(axis=1)]
        hand_mesh = trimesh.Trimesh(vertices=np.asarray(smplh_verts[0]), faces=hand_faces, process=False)
        # palmar contact vertices from camera rays
        body_mesh = trimesh.Trimesh(vertices=np.asarray(smplh_verts[0]), faces=body_faces, process=False)
        vert_normals = np.asarray(body_mesh.vertex_normals)
        palm_normal = np.zeros((len(hand_idx), 3), np.float64)
        offset = 0
        for n in contacting:
            full = np.asarray(RIGHT_HAND_FULL if n == "right" else LEFT_HAND_FULL)
            palmar = np.asarray(RIGHT_HAND_VERTS if n == "right" else LEFT_HAND_VERTS)
            pn = vert_normals[palmar].mean(0); pn /= (np.linalg.norm(pn) + 1e-9)
            palm_normal[offset:offset + len(full)] = pn
            offset += len(full)
        x0 = max(0, int(hand_px_np[:, 0].min())); x1 = min(mask_w, int(hand_px_np[:, 0].max()) + 1)
        y0 = max(0, int(hand_px_np[:, 1].min())); y1 = min(mask_h, int(hand_px_np[:, 1].max()) + 1)
        obj_region = (amodal_arr > 0)
        ray_region = np.zeros_like(obj_region); ray_region[y0:y1, x0:x1] = obj_region[y0:y1, x0:x1]
        ray_ys, ray_xs = np.nonzero(ray_region)
        keep_np = np.zeros(len(hand_verts), dtype=bool)
        n_rays = len(ray_xs)
        if n_rays:
            fx, fy = float(focal_length[0]), float(focal_length[1])
            cx, cy = float(img_w) / 2.0, float(img_h) / 2.0
            dirs = np.stack([(ray_xs - cx) / fx, (ray_ys - cy) / fy, np.ones(n_rays)], axis=1)
            origins = np.zeros((n_rays, 3), np.float64)
            hits, _, hit_tri = hand_mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
            body_verts = np.asarray(smplh_verts[0])
            hand_row = {int(v): i for i, v in enumerate(hand_idx)}
            for k in range(len(hits)):
                tri = hand_faces[int(hit_tri[k])]
                nearest = int(tri[np.argmin(((body_verts[tri] - hits[k]) ** 2).sum(1))])
                row = hand_row[nearest]
                if float(vert_normals[nearest].dot(palm_normal[row])) > 0.0:
                    keep_np[row] = True
        keep = torch.as_tensor(keep_np, device=dev)
        obj_faces_np = obj_faces.cpu().numpy() if torch.is_tensor(obj_faces) else np.asarray(obj_faces)
        obj_np = obj.cpu().numpy()
        hand_verts_np = np.asarray(smplh_verts[0])[hand_idx].astype(np.float32)

        def surface_loss(m, targets):
            """Nearest-surface loss of targets to the object scaled by m."""
            d2 = knn_points(targets.unsqueeze(0), (m * obj_sub).unsqueeze(0), K=1)[0].squeeze(-1)
            return float((F.relu(torch.sqrt(d2.clamp_min(1e-12)) - 0.005) ** 2).mean())

        def pen_grid(scales):
            """Mean squared hand penetration at each scale."""
            queries = np.concatenate([(hand_verts_np / m).astype(np.float32) for m in scales], 0)
            sd, _, _, _ = igl.signed_distance(
                queries, obj_np.astype(np.float32), obj_faces_np.astype(np.int32),
                igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER)
            sd = np.asarray(sd, np.float32).reshape(len(scales), len(hand_verts_np))
            return [float((np.maximum(-(sd[i] * m), 0.0) ** 2).mean())
                    for i, m in enumerate(scales)]

        def grid_search(objective, lo, hi):
            """Coarse-to-fine grid search of objective on [lo, hi)."""
            step = (hi - lo) / 40.0
            best = 0.5 * (lo + hi)
            for _ in range(4):
                grid = np.arange(lo, hi, step)
                vals = objective([float(m) for m in grid])
                best = float(grid[int(np.argmin(vals))])
                lo, hi, step = max(best - step, 0.1 * best), best + step, step / 5.0
            return best

        def max_pen(m):
            """Maximum hand penetration at scale m, in metres."""
            sd, _, _, _ = igl.signed_distance((hand_verts_np).astype(np.float32),
                (obj_np * m).astype(np.float32), obj_faces_np.astype(np.int32),
                igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER)
            return float(np.maximum(-np.asarray(sd, np.float32), 0.0).max())

        # same-ray matching
        obj_px = project_to_image(obj, focal_length, img_w, img_h).detach().cpu().numpy()
        obj_z = obj[:, 2].detach().cpu().numpy()
        obj_tree = cKDTree(obj_px)

        # near or far surface, from the visible mask
        visible = np.asarray(vis_mask) > 127
        if visible.shape != (int(img_h), int(img_w)):
            visible = cv2.resize(visible.astype(np.uint8), (int(img_w), int(img_h)),
                                 interpolation=cv2.INTER_NEAREST) > 0

        def behind_visible(x, y):
            """True if the visible object covers pixel (x, y)."""
            u = min(max(int(round(float(x))), 0), visible.shape[1] - 1)
            v = min(max(int(round(float(y))), 0), visible.shape[0] - 1)
            return bool(visible[v, u])

        def seat_scale(mask):
            """Seating scale of the hand vertices in mask."""
            targets = hand_verts[mask]
            targets_px = hand_px[mask].detach().cpu().numpy()
            targets_z = targets[:, 2].detach().cpu().numpy()
            near, far, hz, rear = [], [], [], []
            for k in range(len(targets_px)):
                cand = obj_tree.query_ball_point(targets_px[k], MATCH_PX)
                if not cand:
                    continue
                cand = np.asarray(cand); cand_z = obj_z[cand]
                near.append(float(cand_z.min())); far.append(float(cand_z.max()))
                hz.append(float(targets_z[k]))
                rear.append(behind_visible(targets_px[k][0], targets_px[k][1]))
            if len(hz) >= 3:
                near_z = np.asarray(near); far_z = np.asarray(far); hand_z = np.asarray(hz)
                use_far = np.asarray(rear, dtype=bool)

                def loss(m):
                    pz = np.where(use_far, far_z * m, near_z * m)
                    return float(((pz - hand_z) ** 2).mean())
                c = max(float(np.median(hand_z)) / max(float(np.median(near_z)), 1e-3), 1e-3)
            else:
                loss = lambda m: surface_loss(m, targets)
                c = max(float(targets[:, 2].median()) / max(float(obj_sub[:, 2].median()), 1e-3), 1e-3)
            N1 = grid_search(lambda scales: [loss(m) for m in scales], c * 0.2, c * 5.0)
            N_seat = N1
            if max_pen(N1) > PEN_THR:
                N_seat = grid_search(lambda scales: [loss(m) + PEN_W * p
                                                     for m, p in zip(scales, pen_grid(scales))],
                                     N1 * 0.5, N1 * 1.6)
            return N_seat

        if int(keep.sum()) >= 3:
            N_seat = seat_scale(keep)
            # fixed pairs for joint refinement
            keep_px = hand_px[keep].detach().cpu().numpy()
            keep_idx = hand_idx[keep.cpu().numpy()]
            pair_hand, pair_obj = [], []
            for k in range(len(keep_px)):
                cand = obj_tree.query_ball_point(keep_px[k], MATCH_PX)
                if not cand:
                    continue
                cand = np.asarray(cand); cand_z = obj_z[cand]
                is_rear = behind_visible(keep_px[k][0], keep_px[k][1])
                pair_obj.append(int(cand[cand_z.argmax()] if is_rear else cand[cand_z.argmin()]))
                pair_hand.append(int(keep_idx[k]))
            if len(pair_hand) >= 3:
                pair_hand_np = np.asarray(pair_hand, np.int64)
                pair_obj_np = np.asarray(pair_obj, np.int64)
                left_set = set(int(i) for i in LEFT_HAND_FULL)
                is_left = np.array([int(h) in left_set for h in pair_hand_np], dtype=bool)
                contact_pairs = {"left": (pair_hand_np[is_left], pair_obj_np[is_left]),
                                 "right": (pair_hand_np[~is_left], pair_obj_np[~is_left])}
            label = f"{int(keep.sum())} palmar contact verts, {len(pair_hand)} same-ray pairs"
        else:
            # fingertip fallback
            tip_idx = np.concatenate([np.asarray(LEFT_FINGERTIPS if n == "left" else RIGHT_FINGERTIPS)
                                      for n in contacting]).astype(int)
            tips = torch.as_tensor(np.asarray(smplh_verts[0])[tip_idx], dtype=torch.float32, device=dev)
            c = max(float(tips[:, 2].median()) / max(float(obj_sub[:, 2].median()), 1e-3), 1e-3)
            N1 = grid_search(lambda scales: [surface_loss(m, tips) for m in scales], c * 0.2, c * 5.0)
            N_seat = N1
            if max_pen(N1) > PEN_THR:
                N_seat = grid_search(lambda scales: [surface_loss(m, tips) + PEN_W * p
                                                     for m, p in zip(scales, pen_grid(scales))],
                                     N1 * 0.5, N1 * 1.6)
            label = f"{int(keep.sum())} palmar contact verts (<3), fingertip fallback"
        obj_verts = (N_seat * obj).cpu().numpy()
        print(f"[seat] {len(contacting)} contacting hand(s): {label} -> scale {N_seat:.4f}", flush=True)
    else:
        obj_verts = (obj_verts_fit * N_depth).detach().cpu().numpy()
        print(f"[seat] no contact -> depth-map scale {N_depth:.4f}", flush=True)

    return det, left_contact, right_contact, obj_verts, contact_pairs, N_depth
