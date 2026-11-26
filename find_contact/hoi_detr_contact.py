"""Match HOI-DETR hand detections and contact links to the SMPL-H hands."""
import numpy as np
import torch

from optimizer.common import LEFT_HAND_FULL, RIGHT_HAND_FULL, project_to_image
from utils.projector_2d import make_square_bbox

CONTACT_PROB_THR = 0.5    # min. hand-object link probability
HAND_COVER_THR = 0.5      # min. coverage of a detected hand box
HAND_BOX_DILATE = 0.10    # SMPL-H hand box dilation per side (fraction)
TARGET_MATCH_THR = 0.1    # min. match score to the target object
WILOR_BOX_SCALE = 1.5     # WiLoR box size / detected box size


def bbox_from_verts_2d(verts_2d):
    """xyxy bbox of 2D points (N, 2)."""
    v = np.asarray(verts_2d, dtype=np.float64).reshape(-1, 2)
    if len(v) == 0:
        return None
    x0, y0 = v.min(0)
    x1, y1 = v.max(0)
    return [float(x0), float(y0), float(x1), float(y1)]


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _iou(a, b):
    """IoU of two xyxy boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _centroid(b):
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def _object_match_score(a, b):
    """Match score of box a to the target box b."""
    iou = _iou(a, b)
    if iou <= 0:
        return 0.0
    aa, ba = _area(a), _area(b)
    scale = min(aa, ba) / max(aa, ba) if max(aa, ba) > 0 else 0.0
    diag = ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5
    cdist = float(np.linalg.norm(_centroid(a) - _centroid(b)))
    center = max(0.0, 1.0 - cdist / (diag + 1e-9))
    return iou * scale * center


def _target_firstobject(record, obj_bbox):
    """Index of the best first-object match; None without obj_bbox, -1 if none."""
    if obj_bbox is None:
        return None
    best_idx, best = None, 0.0
    for i, d in enumerate(record.get("detections", [])):
        if d.get("class_id") != 1:          # first object
            continue
        s = _object_match_score(list(d["box"]), obj_bbox)
        if s > best:
            best, best_idx = s, i
    return best_idx if best >= TARGET_MATCH_THR else -1


def _near_mask(box, mbb):
    """True if box largely overlaps the target-object box mbb."""
    ix = max(0.0, min(box[2], mbb[2]) - max(box[0], mbb[0]))
    iy = max(0.0, min(box[3], mbb[3]) - max(box[1], mbb[1]))
    inter = ix * iy
    if inter <= 0:
        return False
    if inter / max(_area(mbb), 1e-9) >= 0.60:
        return True
    return _iou(box, mbb) >= 0.50


def _detr_hands(record, prob_thr, target_obj_idx=None, obj_bbox=None):
    """HOI-DETR hand detections {idx, box, contact, prob}."""
    dets = record.get("detections", [])
    hf = record.get("hf", [])
    best = {}
    for pair in hf:
        a, b, p = pair["a"], pair["b"], float(pair.get("prob", 1.0))
        if target_obj_idx is not None:
            if target_obj_idx >= 0:
                if b != target_obj_idx:
                    continue
            else:
                if not (0 <= b < len(dets)) or not _near_mask(list(dets[b]["box"]), obj_bbox):
                    continue
        if a not in best or p > best[a]:
            best[a] = p
    hands = []
    for i, d in enumerate(dets):
        if d.get("class_id") != 0:
            continue
        p = best.get(i, 0.0)
        hands.append({"idx": i, "box": list(d["box"]), "contact": p >= prob_thr, "prob": p})
    return hands


def _scale_hands_to(hands, record, img_wh):
    """Rescale hand boxes in place to img_wh."""
    rw, rh = record.get("width"), record.get("height")
    if rw and rh and (abs(rw - img_wh[0]) > 0.5 or abs(rh - img_wh[1]) > 0.5):
        sx, sy = img_wh[0] / float(rw), img_wh[1] / float(rh)
        for h in hands:
            b = h["box"]
            h["box"] = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]


def hand_detections(record, img_wh, prob_thr=0.0, target_obj_idx=None, obj_bbox=None):
    """HOI-DETR hand detections with boxes scaled to img_wh."""
    hands = _detr_hands(record, prob_thr, target_obj_idx=target_obj_idx, obj_bbox=obj_bbox)
    _scale_hands_to(hands, record, img_wh)
    return hands


def _coverage(det_box, smpl_box):
    """Fraction of det_box covered by the dilated SMPL-H hand box."""
    cx = 0.5 * (smpl_box[0] + smpl_box[2]); cy = 0.5 * (smpl_box[1] + smpl_box[3])
    s = 0.5 * max(smpl_box[2] - smpl_box[0], smpl_box[3] - smpl_box[1])
    s *= (1.0 + 2.0 * HAND_BOX_DILATE)
    e = [cx - s, cy - s, cx + s, cy + s]
    ix = max(0.0, min(det_box[2], e[2]) - max(det_box[0], e[0]))
    iy = max(0.0, min(det_box[3], e[3]) - max(det_box[1], e[1]))
    return (ix * iy) / max(_area(det_box), 1e-9)


def assign_hands_to_lr(hands, cam_bboxes):
    """One-to-one left/right assignment. Returns {side: (hand index, coverage) or None}."""
    res = {"left": None, "right": None}
    present = [s for s in ("left", "right") if cam_bboxes.get(s) is not None]
    if not hands or not present:
        return res
    cov = {s: [_coverage(h["box"], cam_bboxes[s]) for h in hands] for s in present}

    if len(present) == 1:
        s = present[0]
        j = int(np.argmax(cov[s]))
        if cov[s][j] >= HAND_COVER_THR:
            res[s] = (j, float(cov[s][j]))
        return res

    n = len(hands)
    lc, rc = _centroid(cam_bboxes["left"]), _centroid(cam_bboxes["right"])
    inplay = [j for j in range(n)
              if max(cov["left"][j], cov["right"][j]) >= HAND_COVER_THR]
    cand = {s: list(inplay) + [None] for s in ("left", "right")}
    best = None
    for jl in cand["left"]:
        for jr in cand["right"]:
            if jl is not None and jr is not None and jl == jr:
                continue
            n_assigned = int(jl is not None) + int(jr is not None)
            dist = 0.0
            if jl is not None:
                dist += float(np.linalg.norm(lc - _centroid(hands[jl]["box"])))
            if jr is not None:
                dist += float(np.linalg.norm(rc - _centroid(hands[jr]["box"])))
            key = (-n_assigned, dist)
            if best is None or key < best[0]:
                best = (key, jl, jr)
    _, jl, jr = best
    if jl is not None:
        res["left"] = (jl, float(cov["left"][jl]))
    if jr is not None:
        res["right"] = (jr, float(cov["right"][jr]))
    return res


def assign_hand_boxes(record, smpl_verts0, focal_length, img_w, img_h):
    """WiLoR crop boxes of the detected hands, left first."""
    hands = hand_detections(record, img_wh=(float(img_w), float(img_h)))
    if not hands:
        print("[hands] no HOI-DETR hand detection", flush=True)
        return [], []
    sv = np.asarray(smpl_verts0)
    # floats keep the projection on the CPU
    fx, fy = float(focal_length[0]), float(focal_length[1])
    iw, ih = float(img_w), float(img_h)
    cam_bb = {}
    for side, hidx in (("left", LEFT_HAND_FULL), ("right", RIGHT_HAND_FULL)):
        p2 = project_to_image(torch.as_tensor(sv[np.asarray(hidx)], dtype=torch.float32), (fx, fy), iw, ih)
        cam_bb[side] = bbox_from_verts_2d(p2.detach().cpu().numpy())
    assign = assign_hands_to_lr(hands, cam_bb)
    boxes, is_right = [], []
    for side, label in (("left", 0), ("right", 1)):   # estimate_hands expects left first
        a = assign[side]
        if a is None:
            continue
        j, _ = a
        b = make_square_bbox(hands[j]["box"], img_w, img_h, scale=WILOR_BOX_SCALE)
        boxes.append(np.array(b, dtype=np.float32))
        is_right.append(label)
    print(f"[hands] {len(hands)} HOI-DETR hand detection(s) -> "
          f"left: {'yes' if 0 in is_right else 'no'}, right: {'yes' if 1 in is_right else 'no'}", flush=True)
    return boxes, is_right


def associate(record, cam_bboxes, img_wh, prob_thr=CONTACT_PROB_THR, obj_bbox=None):
    """Per-hand (contact, prob, coverage, detr_idx) from a HOI-DETR record."""
    out = {"left": (False, 0.0, 0.0, None), "right": (False, 0.0, 0.0, None)}
    target_idx = _target_firstobject(record, obj_bbox)
    hands = hand_detections(record, img_wh=img_wh, prob_thr=prob_thr, target_obj_idx=target_idx,
                            obj_bbox=obj_bbox)
    if not hands:
        return out
    assign = assign_hands_to_lr(hands, cam_bboxes)
    for s in ("left", "right"):
        a = assign[s]
        if a is None:
            continue
        j, cov = a
        h = hands[j]
        out[s] = (bool(h["contact"]), float(h["prob"]), float(cov), int(h["idx"]))
    return out
