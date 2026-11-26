"""WiLoR hands on the SAM 3D Body initialization."""
import gc
import torch
from tqdm import trange

from optimizer.common import split_wrist_twist
from optimizer.wristoptim import WristOptimizer, WristPosOptimizer

WILOR_RESCALE = 2.0         # WiLoR crop enlargement
WILOR_GATE_DEG = 90.0       # max. angle to the SAM 3D Body hand (deg)
TWIST_SPLIT_FRAC = 0.5      # wrist twist fraction moved to the elbow
TWIST_SPLIT_MIN_DEG = 60.0  # min. wrist twist to split (deg)


def estimate_hands(wilor, human, img_cv2, boxes, is_right,
                   left_flag, right_flag, out_smpl_params, out_cam,
                   focal_length_, batch):
    """Insert the WiLoR hands into the SMPL body."""
    left_kpts = None
    right_kpts = None

    wilor.model.to(wilor.device)
    mano_params, mano_kpts = wilor.run(
        img_cv2, boxes, is_right, left_flag, right_flag,
        rescale_factor=WILOR_RESCALE)
    wilor.model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    if left_flag:
        left_kpts = mano_kpts[0]
        if right_flag:
            right_kpts = mano_kpts[1]
    elif right_flag:
        right_kpts = mano_kpts[0]

    # elbow alignment to the WiLoR wrists
    wrist_pos_opt = WristPosOptimizer(
        smpl_params=out_smpl_params, cam=out_cam, focal_length=focal_length_, batch=batch,
        left_wrist_ref=left_kpts, right_wrist_ref=right_kpts, left_flag=left_flag, right_flag=right_flag)
    wrist_pos_opt = wrist_pos_opt.to(human.device)
    optimizer = torch.optim.Adam(
        [p for p in (wrist_pos_opt.Lelbow, wrist_pos_opt.Relbow) if p.requires_grad], lr=1e-2)

    for step in trange(60, desc="Elbow alignment"):
        optimizer.zero_grad()
        loss = wrist_pos_opt.loss()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, out_smpl_params = wrist_pos_opt.forward()

    # MANO -> SMPL-H; body_pose[:, k] is SMPL joint k+1
    global_orient = out_smpl_params['global_orient']
    body_pose = out_smpl_params['body_pose'][:, :21]
    betas = out_smpl_params['betas']
    wrist0 = body_pose[:, [19, 20]].clone()        # SAM 3D Body's own wrists

    spine = global_orient @ body_pose[:, 2] @ body_pose[:, 5] @ body_pose[:, 8]
    left_chain = spine @ body_pose[:, 12] @ body_pose[:, 15] @ body_pose[:, 17]
    right_chain = spine @ body_pose[:, 13] @ body_pose[:, 16] @ body_pose[:, 18]

    adjusted_left = left_chain.transpose(-1, -2) @ mano_params['left_global_orient']
    adjusted_right = right_chain.transpose(-1, -2) @ mano_params['right_global_orient']

    if left_flag:
        body_pose[:, 19] = adjusted_left
    if right_flag:
        body_pose[:, 20] = adjusted_right

    # no box: SAM 3D Body's fingers
    left_hand_pose = mano_params['left_hand_pose']
    right_hand_pose = mano_params['right_hand_pose']
    if not (left_flag and right_flag):
        fitted_left, fitted_right = human.fitted_hand_rotmats()
        if not left_flag:
            left_hand_pose = fitted_left.to(left_hand_pose)
        if not right_flag:
            right_hand_pose = fitted_right.to(right_hand_pose)

    smplh_params = {
        'global_orient': global_orient,
        'body_pose': body_pose,
        'betas': betas,
        'left_hand_pose': left_hand_pose,
        'right_hand_pose': right_hand_pose,
    }

    # WiLoR direction check
    angles = []
    with torch.no_grad():
        rest_joints = human.smplh_model(betas=betas).joints[0]
        bp = smplh_params['body_pose'].clone()
        for k, side, key, flag, i, wj, mj in ((19, 'L', 'left_hand_pose', left_flag, 0, 20, 25),
                                              (20, 'R', 'right_hand_pose', right_flag, 1, 21, 40)):
            if not flag:
                continue
            axis = (rest_joints[mj] - rest_joints[wj]) / (rest_joints[mj] - rest_joints[wj]).norm()
            dir_wilor, dir_body = bp[0, k] @ axis, wrist0[0, k - 19] @ axis
            polar = float(torch.rad2deg(torch.arccos((dir_wilor @ dir_body).clamp(-1.0, 1.0))))
            angles.append(f"{side} {polar:.0f}")
            if polar > WILOR_GATE_DEG:
                bp[:, k] = wrist0[:, k - 19]
                smplh_params[key] = human.fitted_hand_rotmats()[i].to(smplh_params[key])
                if k == 19:
                    left_kpts = None
                else:
                    right_kpts = None
                angles[-1] += " -> SAM 3D Body hand"
        smplh_params['body_pose'] = bp
    print(f"[hands] WiLoR vs SAM 3D Body hand direction (deg): {', '.join(angles) or '-'}", flush=True)

    # wrist refinement to the WiLoR keypoints
    if (left_flag and left_kpts is not None) or (right_flag and right_kpts is not None):
        wrist_opt = WristOptimizer(
            smplh_params=smplh_params, cam=out_cam, focal_length=focal_length_, batch=batch,
            left_kpts=left_kpts, right_kpts=right_kpts, left_flag=left_flag, right_flag=right_flag)
        wrist_opt = wrist_opt.to(human.device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, wrist_opt.parameters()), lr=3e-2)

        for step in trange(100, desc="Wrist rotation"):
            optimizer.zero_grad()
            loss = wrist_opt.loss()
            loss.backward()
            optimizer.step()
            wrist_opt.project()

            with torch.no_grad():
                smplh_params = wrist_opt.forward()

    # wrist twist split
    with torch.no_grad():
        rest_joints = human.smplh_model(betas=smplh_params['betas']).joints[0]
        axes = {19: (17, (rest_joints[20] - rest_joints[18]) / (rest_joints[20] - rest_joints[18]).norm()),
                20: (18, (rest_joints[21] - rest_joints[19]) / (rest_joints[21] - rest_joints[19]).norm())}
        axes = {w: v for w, v in axes.items() if (left_flag if w == 19 else right_flag)}
        bp = smplh_params['body_pose'].clone()
        twist = split_wrist_twist(bp, axes, frac=TWIST_SPLIT_FRAC, min_deg=TWIST_SPLIT_MIN_DEG)
        smplh_params['body_pose'] = bp
    print("[hands] wrist twist about the forearm (deg): "
          + ", ".join(f"{'L' if w == 19 else 'R'} {t:+.0f}" for w, t in twist.items()), flush=True)

    output_vertices, _, output_cam_trans = human.run_body_model(
        human.smplh_model, smplh_params, out_cam, batch)

    smplh_verts = (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()

    return smplh_params, smplh_verts, output_cam_trans, left_kpts, right_kpts


def fitted_hands_only(human, out_smpl_params, out_cam, batch):
    """SMPL-H body with SAM 3D Body's fingers, without hand boxes."""
    fitted_left, fitted_right = human.fitted_hand_rotmats()
    smplh_params = {'global_orient': out_smpl_params['global_orient'],
                    'body_pose': out_smpl_params['body_pose'][:, :21],
                    'betas': out_smpl_params['betas'],
                    'left_hand_pose': fitted_left, 'right_hand_pose': fitted_right}
    with torch.no_grad():
        output_vertices, _, output_cam_trans = human.run_body_model(
            human.smplh_model, smplh_params, out_cam, batch)
    smplh_verts = (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
    print("[hands] no hand box: the SMPL-H body keeps SAM 3D Body's fitted fingers", flush=True)
    return smplh_params, smplh_verts, output_cam_trans
