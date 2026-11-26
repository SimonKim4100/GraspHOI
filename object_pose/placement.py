"""Metric object placement in the camera frame."""
import numpy as np
import torch
from tqdm import trange

from object_pose.foundationpose_register import register_object_pose
from object_pose.layout import LAYOUT_IOU_GATE, fit_iou, place_from_layout, refit_and_pull
from object_pose.ppd_depth import predict_depth_amodal, park_cpu as _ppd_park
from object_pose.scale import extent_fallback_rescale, sphere_rescale
from optimizer.objoptim import ObjectOptimizer, iou_of

OBJ_PATIENCE = 75      # early stop patience
OBJ_MIN_STEPS = 150    # no early stop before this step


def fit_context(obj_faces, focal_length_, batch, smplh_verts, amodal_sil, left_flag, right_flag):
    """ObjectOptimizer arguments independent of the starting pose."""
    return {'img_w': int(batch['img_size'][0][1]), 'img_h': int(batch['img_size'][0][0]),
            'focal_length': (focal_length_[0], focal_length_[0]), 'obj_faces': obj_faces,
            'smplh_verts': smplh_verts, 'hands_detected': [left_flag, right_flag], 'amodal_mask': amodal_sil}


def fit_pose(ctx, verts, R, t, from_pose, tag):
    """Fit the object pose with L_obj. Returns (model, fitted verts, IoU)."""
    cfg = dict(ctx, og_obj_verts=verts, R_mat=R, init_from_pose=from_pose, translation=t)
    m = ObjectOptimizer(**cfg)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    best_v = float('inf'); best_sd = None; since_best = 0
    for step in trange(300, desc=f"[objoptim] {tag}"):
        opt.zero_grad()
        scale_loss, pos_loss, sil_loss, cont_loss = m.all_losses()
        # contour weight ramp
        w_cont = 0.3 + (1.0 - 0.3) * (step / 299.0)
        total = 10 * (scale_loss + pos_loss) + sil_loss + w_cont * cont_loss
        # keep the best state
        v = float(total)
        if v < best_v:
            best_v = v; best_sd = {k: x.clone().detach() for k, x in m.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if step >= OBJ_MIN_STEPS and since_best >= OBJ_PATIENCE:
                break
        total.backward()
        for p in m.parameters():
            if p.grad is not None:
                torch.nn.utils.clip_grad_norm_([p], 1e4)
        opt.step()
    m.load_state_dict(best_sd)
    with torch.no_grad():
        vv = m().detach()
    return m, vv, iou_of(m, vv)


def fit_from_verts(V, ctx, tag):
    """Fit from camera-frame vertices. Returns (model, fitted verts, IoU, centred verts)."""
    V = np.asarray(V, np.float32)
    c = V.mean(0)
    m, vv, iou = fit_pose(ctx, V - c, np.eye(3, dtype=np.float32), c, True, tag)
    return m, vv, iou, V - c


def place_fallback(mesh, obj_faces, ctx, cam_int, img_cv2, amodal_on_white, amodal_sil, vis_mask, depth_map,
                   device='cuda'):
    """FoundationPose placement. Returns (model, verts, IoU, scaled verts, info)."""
    fx_int = float(cam_int[0][0])
    og_orig = np.asarray(mesh.vertices, dtype=np.float32).copy()
    info = {}

    # as-if-unoccluded RGB-D for FoundationPose
    depth_fp = predict_depth_amodal(amodal_on_white, depth_map, amodal_sil, vis_mask)
    _ppd_park()
    bgr_fp = img_cv2.copy()
    in_sil = amodal_sil > 127
    bgr_fp[in_sil] = amodal_on_white[..., ::-1][in_sil]        # RGB -> BGR
    F = obj_faces.cpu().numpy() if torch.is_tensor(obj_faces) else np.asarray(obj_faces)

    def fp_and_fit(verts, tag):
        mesh.vertices = verts
        pose = register_object_pose(mask=amodal_sil, mesh=mesh, bgr=bgr_fp, K=cam_int, depth=depth_fp)
        R = pose[:3, :3].astype(np.float32); t = pose[:3, 3].astype(np.float32)
        # t = 0: no valid depth inside the mask
        if not float(np.linalg.norm(t)) > 1e-6:
            print("[fp] registration failed -> mask-centered init", flush=True)
            info.update(fp_ok=False, fit=True)
            return fit_pose(ctx, verts, R, t, False, tag)
        V, pull = refit_and_pull((R @ np.asarray(verts, np.float32).T).T + t, F, cam_int, amodal_sil,
                                 vis_mask, depth_map, device=device)
        c = V.mean(0)
        m, vv, iou = fit_pose(ctx, V - c, np.eye(3, dtype=np.float32), c, True, tag)
        i0, i1 = fit_iou(V, F, cam_int, amodal_sil, vis_mask), fit_iou(vv, F, cam_int, amodal_sil, vis_mask)
        info.update(fp_ok=True, iou_refit=pull['iou'], N_d=pull['N'], iou_pulled=i0, iou_fit=i1, fit=i1 > i0)
        if i1 > i0:
            return m, vv, iou
        vt = torch.as_tensor(V, dtype=torch.float32, device=vv.device)
        return (lambda _t=vt: _t), vt, i0

    verts_sph, ms_sph = sphere_rescale(og_orig, depth_map, vis_mask, cam_int)
    if verts_sph is not None:
        info.update(scale='sphere', mscale=ms_sph)
        model, verts_2d, iou_fit = fp_and_fit(verts_sph, "sphere")
        og_obj_verts = verts_sph
    else:
        verts_fb, ms_fb = extent_fallback_rescale(og_orig, amodal_sil, depth_map, fx_int)
        print(f"[scale] sphere scale unusable -> extent scale {ms_fb:.3f}", flush=True)
        info.update(scale='extent', mscale=ms_fb)
        model, verts_2d, iou_fit = fp_and_fit(verts_fb, "extent")
        og_obj_verts = verts_fb
    mesh.vertices = og_obj_verts
    return model, verts_2d, iou_fit, og_obj_verts, info


def place_object(mesh, layout, obj_faces, cam_int, focal_length_, batch, img_cv2, amodal_on_white, amodal_sil,
                 vis_mask, depth_map, smplh_verts, left_flag, right_flag, device='cuda'):
    """Metric placement of the mesh: SAM 3D layout, else FoundationPose."""
    ctx = fit_context(obj_faces, focal_length_, batch, smplh_verts, amodal_sil, left_flag, right_flag)
    lay = place_from_layout(mesh, layout, cam_int, amodal_sil, vis_mask, depth_map, device=device)
    info = dict(iou_layout_raw=lay['iou_raw'], iou_layout=lay['iou'], N_d=lay.get('N'), why=lay.get('why'),
                iou_layout_pulled=None, iou_layout_fit=None, iou_fp=None, iou_layout_kept=None)

    # object fit from the placed layout
    lfit = None
    if lay.get('verts') is not None:
        lfit = fit_from_verts(lay['verts'], ctx, "layout")
        info['iou_layout_pulled'] = fit_iou(lay['verts'], obj_faces, cam_int, amodal_sil, vis_mask)
        info['iou_layout_fit'] = fit_iou(lfit[1], obj_faces, cam_int, amodal_sil, vis_mask)
        if not info['iou_layout_fit'] > info['iou_layout_pulled']:
            lfit = None

    use_layout = lay['use_layout']
    fp = None
    if not use_layout:
        model, verts_2d, _, og_obj_verts, fp = place_fallback(
            mesh, obj_faces, ctx, cam_int, img_cv2, amodal_on_white, amodal_sil, vis_mask, depth_map,
            device=device)
        if lay.get('verts') is not None:
            # keep the better of layout and fallback
            info['iou_fp'] = fit_iou(verts_2d, obj_faces, cam_int, amodal_sil, vis_mask)
            info['iou_layout_kept'] = fit_iou(lfit[1] if lfit is not None else lay['verts'], obj_faces,
                                              cam_int, amodal_sil, vis_mask)
            use_layout = info['iou_layout_kept'] > info['iou_fp']
    if use_layout and lfit is not None:
        model, verts_2d, _, og_obj_verts = lfit
        mesh.vertices = og_obj_verts
    elif use_layout:
        lv = torch.as_tensor(lay['verts'], dtype=torch.float32, device=device)
        model = (lambda _t=lv: _t)        # ObjectOptimizer stand-in
        verts_2d = lv
        og_obj_verts = lay['verts']
        mesh.vertices = og_obj_verts
    info.update(path='layout' if use_layout else 'foundationpose',
                fit=(lfit is not None) if use_layout else fp['fit'],
                iou_final=fit_iou(verts_2d, obj_faces, cam_int, amodal_sil, vis_mask), fp=fp)
    print(_placement_line(info), flush=True)
    return model, verts_2d, og_obj_verts, info


def _placement_line(info):
    """One-line summary of place_object's info."""
    def f(x):
        return 'n/a' if x is None else f'{x:.3f}'

    s = (f"[placement] {info['path']}{' + L_obj' if info['fit'] else ''}: "
         f"layout IoU {f(info['iou_layout_raw'])} -> re-fit {f(info['iou_layout'])} (gate {LAYOUT_IOU_GATE})")
    if info['N_d'] is not None:
        s += f", N_d {info['N_d']:.4f}"
    if info['iou_layout_fit'] is not None:
        s += f", L_obj {f(info['iou_layout_pulled'])} -> {f(info['iou_layout_fit'])}"
    fp = info['fp']
    if fp is not None:
        s += f"; FoundationPose ({info['why']}, {fp['scale']} scale {fp['mscale']:.3f}"
        if fp.get('fp_ok'):
            s += (f", re-fit {f(fp['iou_refit'])}, N_d {f(fp['N_d'])}, L_obj {f(fp['iou_pulled'])} -> "
                  f"{f(fp['iou_fit'])}")
        else:
            s += ", registration failed"
        s += ")"
        if info['iou_fp'] is not None:
            s += f", keep-better: FoundationPose {f(info['iou_fp'])} vs layout {f(info['iou_layout_kept'])}"
    return s + f"; final IoU {info['iou_final']:.3f}"
