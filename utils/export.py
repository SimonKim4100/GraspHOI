"""Per-sample export of arrays, contact record, meshes and renders."""
import os
import cv2
import json
import trimesh
import numpy as np
import torch

from utils.renderer_pyrd import Renderer


def _np(x, dtype):
    """Tensor or array -> numpy array of dtype."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def export_sample(output_img_folder, fname, img_ext, img_cv2, cam_int, smplh_params, transl,
                  smplh_verts, smplh_joints, smplh_faces, obj_verts, obj_faces, obj_visual,
                  amodal_on_white, det, left_flag, right_flag, focal_length, img_w, img_h):
    """Write every per-sample output into <output_img_folder>/<fname>/."""
    obj_faces = obj_faces.cpu().numpy() if torch.is_tensor(obj_faces) else np.asarray(obj_faces)

    sample_out = os.path.join(output_img_folder, fname)
    os.makedirs(sample_out, exist_ok=True)

    # camera frame (OpenCV axes, metres)
    np.savez_compressed(
        os.path.join(sample_out, f"{fname}_result.npz"),
        smplh_vertices=_np(smplh_verts[0], np.float32),
        smplh_joints=_np(smplh_joints, np.float32),
        smplh_faces=_np(smplh_faces, np.int64),
        global_orient=_np(smplh_params['global_orient'], np.float32).reshape(1, 1, 3, 3),
        body_pose=_np(smplh_params['body_pose'], np.float32).reshape(1, 21, 3, 3),
        left_hand_pose=_np(smplh_params['left_hand_pose'], np.float32).reshape(1, 15, 3, 3),
        right_hand_pose=_np(smplh_params['right_hand_pose'], np.float32).reshape(1, 15, 3, 3),
        betas=_np(smplh_params['betas'], np.float32).reshape(1, -1),
        transl=_np(transl, np.float32).reshape(3),
        object_vertices=_np(obj_verts, np.float32),
        object_faces=obj_faces.astype(np.int64),
        object_vertex_colors=np.asarray(obj_visual.vertex_colors, dtype=np.uint8),
        K=_np(cam_int, np.float32).reshape(3, 3),
        image_size=np.array([int(img_w), int(img_h)], dtype=np.int32),
        left_hand=np.array(bool(left_flag)), right_hand=np.array(bool(right_flag)))

    def _hand_rec(detected, det_tuple):
        contact, prob, coverage, detr_idx = det_tuple
        return {"detected": bool(detected), "contact": bool(contact),
                "contact_prob": float(prob),
                "assoc_iom": float(coverage),
                "detr_idx": (None if detr_idx is None else int(detr_idx))}
    contact_info = {
        "left_hand":  _hand_rec(left_flag,  det["left"]),
        "right_hand": _hand_rec(right_flag, det["right"]),
    }
    with open(os.path.join(sample_out, f"{fname}_contact.json"), "w") as _fh:
        json.dump(contact_info, _fh, indent=2)

    # camera frame -> upright (y up)
    R_flip = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])

    obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces,
                               visual=obj_visual, process=False)

    smplh_up = trimesh.Trimesh(vertices=smplh_verts[0], faces=smplh_faces, process=False)
    smplh_up.visual.vertex_colors = np.array([185, 185, 210, 255], dtype=np.uint8)
    smplh_up.apply_transform(R_flip)
    obj_up = obj_mesh.copy()
    obj_up.apply_transform(R_flip)

    smplh_up.export(f"{sample_out}/{fname}_smplh.glb", include_normals=True)
    obj_up.export(f"{sample_out}/{fname}_object.glb", include_normals=True)
    # a Scene keeps the object's vertex colors
    combined_scene = trimesh.Scene()
    combined_scene.add_geometry(smplh_up, geom_name='human')
    combined_scene.add_geometry(obj_up, geom_name='object')
    combined_scene.export(f"{sample_out}/{fname}_mesh.glb", include_normals=True)

    cv2.imwrite(f"{sample_out}/{fname}_amodal.png", amodal_on_white[:, :, ::-1])

    renderer = Renderer(focal_length=focal_length[0], img_w=img_w, img_h=img_h)
    front_view = renderer.render_front_view_textured(
        smplh_verts[0], smplh_faces, obj_mesh, bg_img_bgr=img_cv2.copy())

    white_bg = np.ones_like(img_cv2) * 255
    front_view_white_bg = renderer.render_front_view_textured(
        smplh_verts[0], smplh_faces, obj_mesh, bg_img_bgr=white_bg)

    cv2.imwrite(f"{sample_out}/{fname}_overlay{img_ext}", front_view)
    cv2.imwrite(f"{sample_out}/{fname}_whitebg{img_ext}", front_view_white_bg)
    renderer.delete()   # before the side-view renderer

    side_triplet = renderer.render_side_triplet(smplh_up, obj_up)
    cv2.imwrite(f"{sample_out}/{fname}_sides{img_ext}", side_triplet)
    print(f"[export] {sample_out}", flush=True)
