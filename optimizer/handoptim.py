"""Standard stage of the joint refinement: arms, fingers and object scale."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np
import smplx
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points

from human_recon.constants import SMPLH_MODEL_PATH
from human_recon.constants import NUM_BETAS
from utils.projector_2d import project_joints_to_2d
from optimizer.common import (
    run_body_model, select_contact_vertices, first_or_empty,
    geodesic_angle, angle_barrier, reg_distance,
    TunnelGuard, FINGER_CHAINS, ARM_JOINTS, anchor_drift_loss, arm_limit_loss,
    TUNNEL_W, ANCHOR_W, ANCHOR_TOL_HAND, ANCHOR_TOL_ARM, ARM_CAPS_DEG, ARM_CAP_W,
    LEFT_HAND_VERTS, RIGHT_HAND_VERTS,
)

PEN_MARGIN = 0.002      # metres
CONTACT_MARGIN = 0.0    # metres


class HandOptimizer(nn.Module):
    """Arm and finger rotations of the enabled hands, and the object scale."""

    def __init__(self, *, smplh_params, obj_verts, obj_faces, cam, focal_length, batch, smplh_verts,
                 left_kpts, right_kpts, amodal_mask,
                 left_flag: bool, right_flag: bool,
                 contact_pairs=None,
                 reg_w_arm: float = 0.5,
                 reg_w_hand: float = 0.3,
                 angle_barrier_deg: float = 3.0,
                 angle_barrier_pow: int = 4,
                 angle_barrier_scale: float = 20.0):
        super().__init__()
        self.device = torch.device('cuda')

        self.left_enabled  = bool(left_flag)
        self.right_enabled = bool(right_flag)
        if not (self.left_enabled or self.right_enabled):
            raise ValueError("Nothing to optimize: both sides disabled or missing refs.")

        self.obj_verts = obj_verts
        self.obj_faces = obj_faces
        self.smplh_model = smplx.SMPLHLayer(model_path=SMPLH_MODEL_PATH, num_betas=NUM_BETAS).to(self.device)
        self.smplh_params = {k: v.to(self.device) for k, v in smplh_params.items()}
        self.cam = cam.to(self.device)
        self.fl  = focal_length.to(self.device)
        self.batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        self.img_h, self.img_w = self.batch['img_size'][0].tolist()

        self.left_verts_idx = LEFT_HAND_VERTS
        self.right_verts_idx = RIGHT_HAND_VERTS

        output_verts, output_joints, output_cam_trans = run_body_model(self.smplh_model, self.smplh_params,
                                                                       self.cam, self.batch)
        self.trans = output_cam_trans
        init_joints_2d = project_joints_to_2d(output_joints, output_cam_trans, self.fl[0], self.img_w,
                                              self.img_h)

        with torch.no_grad():
            obj_faces_np = obj_faces.detach().long().cpu().numpy()
        self.obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces_np, process=False)
        self.tunnel = TunnelGuard(self.obj_mesh.vertices, self.obj_mesh.faces)
        self.tunnel_sides = [s for s, e in (("left", self.left_enabled), ("right", self.right_enabled)) if e]
        self.n_tunnel = 0

        # amodal silhouette (H, W)
        amodal_mask = np.asarray(amodal_mask)
        if amodal_mask.dtype != np.uint8:
            raise TypeError(f"amodal_mask must be the uint8 (0 / 255) S_a, got {amodal_mask.dtype}")
        self.object_mask = torch.from_numpy(amodal_mask >= 128).to(self.device)

        self.body_pose = self.smplh_params['body_pose'].detach().clone()

        # WiLoR 2D keypoints without the wrist
        self.register_buffer('left_ref',  torch.empty(0, 2, dtype=torch.float32, device=self.device))
        self.register_buffer('right_ref', torch.empty(0, 2, dtype=torch.float32, device=self.device))
        if self.left_enabled:
            self.left_ref = torch.as_tensor(left_kpts[1:], dtype=torch.float32, device=self.device)
        if self.right_enabled:
            self.right_ref = torch.as_tensor(right_kpts[1:], dtype=torch.float32, device=self.device)

        # SMPL-H finger joints
        self.left_idxs  = list(range(22, 22 + 15))
        self.right_idxs = list(range(22 + 15, 22 + 30))

        # shoulder, elbow, wrist
        left_R  = matrix_to_axis_angle(self.body_pose[:, [15, 17, 19]])
        right_R = matrix_to_axis_angle(self.body_pose[:, [16, 18, 20]])
        left_hand = matrix_to_axis_angle(self.smplh_params['left_hand_pose'])
        right_hand = matrix_to_axis_angle(self.smplh_params['right_hand_pose'])

        self.left_rot  = nn.Parameter(left_R.clone(),  requires_grad=self.left_enabled)
        self.right_rot = nn.Parameter(right_R.clone(), requires_grad=self.right_enabled)
        self.left_hand_rot  = nn.Parameter(left_hand.clone(),  requires_grad=self.left_enabled)
        self.right_hand_rot = nn.Parameter(right_hand.clone(), requires_grad=self.right_enabled)

        # finger rotation barrier
        self.ang_thresh_rad    = torch.deg2rad(torch.tensor(angle_barrier_deg, dtype=torch.float32,
                                                            device=self.device))
        self.ang_barrier_pow   = int(angle_barrier_pow)
        self.ang_barrier_scale = float(angle_barrier_scale)

        self.register_buffer('init_left_arm_R',  self.body_pose[:, [15, 17, 19]].detach().clone())
        self.register_buffer('init_right_arm_R', self.body_pose[:, [16, 18, 20]].detach().clone())
        self.register_buffer('init_left_hand_R',  self.smplh_params['left_hand_pose'].detach().clone())
        self.register_buffer('init_right_hand_R', self.smplh_params['right_hand_pose'].detach().clone())

        self.reg_w_arm = float(reg_w_arm)
        self.reg_w_hand = float(reg_w_hand)

        # arm anchors of the keypoint loss
        self.init_left_arms = init_joints_2d[:, [16, 18, 20]]
        self.init_right_arms = init_joints_2d[:, [17, 19, 21]]
        # 2D trust region
        hand_j = sorted({j for s in self.tunnel_sides for c in FINGER_CHAINS[s] for j in c})
        arm_j = sorted({j for s in self.tunnel_sides for j in ARM_JOINTS[s]})
        self.anchor_idx = hand_j + arm_j
        self.anchor_tol = torch.tensor([ANCHOR_TOL_HAND] * len(hand_j) + [ANCHOR_TOL_ARM] * len(arm_j),
                                       device=self.device)
        # arm rotation caps
        self.arm_caps = torch.deg2rad(torch.tensor(ARM_CAPS_DEG, device=self.device))
        self.register_buffer("anchor_2d", init_joints_2d.detach().clone())

        # contact candidates
        sel = select_contact_vertices(
            output_verts, output_cam_trans, self.fl[0], self.img_w, self.img_h,
            self.left_verts_idx, self.right_verts_idx, self.object_mask,
            self.left_enabled, self.right_enabled
        )
        self.init_left_idx  = first_or_empty(sel.get("left_idx_in", []),  self.device) \
                            if self.left_enabled else torch.empty(0, dtype=torch.long, device=self.device)
        self.init_right_idx = first_or_empty(sel.get("right_idx_in", []), self.device) \
                            if self.right_enabled else torch.empty(0, dtype=torch.long, device=self.device)

        # fallback: nearest hand verts
        sv = np.asarray(smplh_verts[0] if not torch.is_tensor(smplh_verts)
                        else smplh_verts[0].detach().cpu().numpy(), dtype=np.float32)
        ov_sub = np.asarray(self.obj_mesh.vertices, dtype=np.float32)[::5]

        def _ensure_candidates(idx_t, hand_list, name):
            if idx_t.numel() >= 8:
                return idx_t
            hl = np.asarray(hand_list)
            dmin = np.linalg.norm(sv[hl][:, None, :] - ov_sub[None], axis=2).min(1)
            near = hl[np.argsort(dmin)[:20]]
            print(f"[{name}] {int(idx_t.numel())} hand verts inside the object mask -> "
                  f"using the 20 nearest to the object (min {dmin.min()*100:.1f} cm)", flush=True)
            return torch.as_tensor(near, dtype=torch.long, device=self.device)

        if self.left_enabled:
            self.init_left_idx = _ensure_candidates(self.init_left_idx, self.left_verts_idx, "hand-L")
        if self.right_enabled:
            self.init_right_idx = _ensure_candidates(self.init_right_idx, self.right_verts_idx, "hand-R")

        # object scale about the camera origin
        self.move_coeff = nn.Parameter(torch.tensor(1.0, device=self.device), requires_grad=True)

        if self.left_enabled:
            self.register_buffer('left_indices_tensor',
                torch.tensor(self.left_verts_idx, dtype=torch.long, device=self.device))
        if self.right_enabled:
            self.register_buffer('right_indices_tensor',
                torch.tensor(self.right_verts_idx, dtype=torch.long, device=self.device))

        if self.left_enabled:
            left_grid, left_origin, left_voxel = self.mesh_to_sdf_grid_single_hand(
                "left", voxel_size=0.001, pad=0.0)
            self.register_buffer("left_sdf_grid", left_grid)
            self.register_buffer("left_sdf_origin", left_origin)
            self.left_sdf_voxel = float(left_voxel) if left_voxel is not None else None
        else:
            self.register_buffer("left_sdf_grid", None)
            self.register_buffer("left_sdf_origin", None)
            self.left_sdf_voxel = None

        if self.right_enabled:
            right_grid, right_origin, right_voxel = self.mesh_to_sdf_grid_single_hand(
                "right", voxel_size=0.001, pad=0.0)
            self.register_buffer("right_sdf_grid", right_grid)
            self.register_buffer("right_sdf_origin", right_origin)
            self.right_sdf_voxel = float(right_voxel) if right_voxel is not None else None
        else:
            self.register_buffer("right_sdf_grid", None)
            self.register_buffer("right_sdf_origin", None)
            self.right_sdf_voxel = None

        self.register_buffer('_obj_verts_t', torch.as_tensor(
            np.asarray(self.obj_mesh.vertices, dtype=np.float32),
            device=self.device, dtype=torch.float32
        ))

        # fixed contact pairs
        empty = np.empty(0, np.int64)
        for side, pfx in (("left", "l"), ("right", "r")):
            ph, po = (contact_pairs or {}).get(side, (empty, empty))
            self.register_buffer(f'pair_{pfx}_hand',
                torch.as_tensor(np.asarray(ph, dtype=np.int64), device=self.device))
            self.register_buffer(f'pair_{pfx}_obj',
                torch.as_tensor(np.asarray(po, dtype=np.int64), device=self.device))

    @staticmethod
    def sample_sdf_world(points_world, sdf_grid, origin_xyz, voxel_size, align_corners=True):
        """Sample an SDF grid (1, 1, Nx, Ny, Nz) at world points (N, 3)."""
        device = sdf_grid.device
        dtype  = sdf_grid.dtype
        points_world = points_world.to(device=device, dtype=dtype)
        origin_xyz   = origin_xyz.to(device=device, dtype=dtype)

        Nx, Ny, Nz = sdf_grid.shape[-3:]
        g = (points_world - origin_xyz) / voxel_size
        gx, gy, gz = g.unbind(-1)
        nx = 2.0 * (gx / (Nx - 1)) - 1.0
        ny = 2.0 * (gy / (Ny - 1)) - 1.0
        nz = 2.0 * (gz / (Nz - 1)) - 1.0
        # grid is stored as (D, H, W) = (Nx, Ny, Nz)
        grid = torch.stack([nz, ny, nx], dim=-1).view(1, -1, 1, 1, 3)

        vals = F.grid_sample(
            sdf_grid, grid.to(dtype).to(device),
            mode='bilinear', padding_mode='border', align_corners=align_corners
        )
        return vals.view(-1)

    def sample_sdf_object_side(self, points_world, side):
        """SDF of the scaled object."""
        s = self.move_coeff.clamp(0.95, 1.05)
        q = points_world / s

        if side == "left":
            vals = self.sample_sdf_world(q, self.left_sdf_grid, self.left_sdf_origin, self.left_sdf_voxel)
        elif side == "right":
            vals = self.sample_sdf_world(q, self.right_sdf_grid, self.right_sdf_origin, self.right_sdf_voxel)
        else:
            raise ValueError(f"Invalid side: {side}")
        return vals * s

    def mesh_to_sdf_grid_single_hand(self, side, voxel_size=0.001, pad=0.0, chunk=2_000_000):
        """Object SDF grid over the hand and object bbox overlap; Nones if disjoint."""
        smplh_params, _ = self.forward()
        output_verts, _, _ = run_body_model(self.smplh_model, smplh_params, self.cam, self.batch)
        output_verts = (output_verts + self.trans.unsqueeze(1)).detach().cpu().numpy()[0]

        if side == "left":
            hand_indices = self.left_verts_idx
        elif side == "right":
            hand_indices = self.right_verts_idx
        else:
            raise ValueError(f"Invalid side: {side}")

        hand_verts = output_verts[hand_indices]
        hand_min = hand_verts.min(axis=0).astype(np.float32)
        hand_max = hand_verts.max(axis=0).astype(np.float32)
        hand_center = (hand_min + hand_max) / 2
        hand_size = hand_max - hand_min

        expanded_size = hand_size * 2.0
        hand_bbox_min = hand_center - expanded_size / 2
        hand_bbox_max = hand_center + expanded_size / 2

        obj_min, obj_max = self.obj_mesh.bounds
        obj_min = obj_min.astype(np.float32, copy=True)
        obj_max = obj_max.astype(np.float32, copy=True)
        obj_min -= pad
        obj_max += pad

        mn = np.maximum(hand_bbox_min, obj_min)
        mx = np.minimum(hand_bbox_max, obj_max)

        if np.any(mn >= mx):
            print(f"[hand-optim] WARNING: object bbox does not overlap {side} hand bbox "
                  f"(gap {np.round(np.maximum(mn - mx, 0.0), 3)}); penetration term disabled "
                  f"for this hand", flush=True)
            return None, None, None
        # bound the grid size
        total = np.prod(np.floor((mx - mn).astype(np.float64) / voxel_size) + 1)
        if total > 64_000_000:
            voxel_size = float(voxel_size * (total / 64_000_000) ** (1.0 / 3.0))
            print(f"[hand-optim] WARNING: {side} SDF grid too large -> voxel coarsened to "
                  f"{voxel_size:.4f} m", flush=True)
        spans = (mx - mn) / voxel_size
        Nx, Ny, Nz = (np.floor(spans) + 1).astype(int)
        xs = np.linspace(mn[0], mx[0], Nx, dtype=np.float32)
        ys = np.linspace(mn[1], mx[1], Ny, dtype=np.float32)
        zs = np.linspace(mn[2], mx[2], Nz, dtype=np.float32)

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        sdf = np.empty((pts.shape[0],), dtype=np.float32)

        obj_verts_np = self.obj_verts
        if torch.is_tensor(obj_verts_np):
            obj_verts_np = obj_verts_np.detach().cpu().numpy()
        obj_verts_np = np.asarray(obj_verts_np, dtype=np.float64)

        obj_faces_np = self.obj_faces
        if torch.is_tensor(obj_faces_np):
            obj_faces_np = obj_faces_np.detach().cpu().numpy()
        obj_faces_np = np.asarray(obj_faces_np, dtype=np.int32)

        # negative inside the object
        import igl
        for a in range(0, pts.shape[0], chunk):
            b = min(a + chunk, pts.shape[0])
            s, _, _, _ = igl.signed_distance(
                pts[a:b], obj_verts_np, obj_faces_np, igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER
            )
            sdf[a:b] = np.asarray(s, np.float32).ravel()

        sdf = sdf.reshape(Nx, Ny, Nz)
        sdf_grid = torch.from_numpy(sdf).unsqueeze(0).unsqueeze(0).to(self.device)
        origin_xyz = torch.tensor(mn.astype(np.float32), device=self.device)
        return sdf_grid, origin_xyz, float(voxel_size)

    def forward(self):
        new_bp = self.body_pose.clone()
        new_left_hand  = self.smplh_params['left_hand_pose'].clone()
        new_right_hand = self.smplh_params['right_hand_pose'].clone()

        if self.left_enabled:
            L_arm = axis_angle_to_matrix(self.left_rot)
            new_bp[:, 15] = L_arm[:, 0]
            new_bp[:, 17] = L_arm[:, 1]
            new_bp[:, 19] = L_arm[:, 2]
            new_left_hand = axis_angle_to_matrix(self.left_hand_rot)

        if self.right_enabled:
            R_arm = axis_angle_to_matrix(self.right_rot)
            new_bp[:, 16] = R_arm[:, 0]
            new_bp[:, 18] = R_arm[:, 1]
            new_bp[:, 20] = R_arm[:, 2]
            new_right_hand = axis_angle_to_matrix(self.right_hand_rot)

        self.smplh_params['body_pose']       = new_bp
        self.smplh_params['left_hand_pose']  = new_left_hand
        self.smplh_params['right_hand_pose'] = new_right_hand
        return self.smplh_params, self.move_coeff

    def regularization_loss(self):
        """Distance to the initial rotations, finger barrier and arm caps."""
        loss = torch.zeros((), device=self.device, dtype=self.body_pose.dtype)

        if self.left_enabled:
            R_arm_L  = axis_angle_to_matrix(self.left_rot)
            R0_arm_L = self.init_left_arm_R
            base_armL = reg_distance(R_arm_L, R0_arm_L).mean()
            loss += self.reg_w_arm * base_armL

            R_hand_L  = axis_angle_to_matrix(self.left_hand_rot)
            R0_hand_L = self.init_left_hand_R
            base_handL = reg_distance(R_hand_L, R0_hand_L).mean()
            ang_handL  = geodesic_angle(R_hand_L, R0_hand_L)
            bar_handL  = angle_barrier(ang_handL, self.ang_thresh_rad, self.ang_barrier_pow).mean()
            loss += self.reg_w_hand * base_handL + (self.reg_w_hand * self.ang_barrier_scale) * bar_handL

        if self.right_enabled:
            R_arm_R  = axis_angle_to_matrix(self.right_rot)
            R0_arm_R = self.init_right_arm_R
            base_armR = reg_distance(R_arm_R, R0_arm_R).mean()
            loss += self.reg_w_arm * base_armR

            R_hand_R  = axis_angle_to_matrix(self.right_hand_rot)
            R0_hand_R = self.init_right_hand_R
            base_handR = reg_distance(R_hand_R, R0_hand_R).mean()
            ang_handR  = geodesic_angle(R_hand_R, R0_hand_R)
            bar_handR  = angle_barrier(ang_handR, self.ang_thresh_rad, self.ang_barrier_pow).mean()
            loss += self.reg_w_hand * base_handR + (self.reg_w_hand * self.ang_barrier_scale) * bar_handR

        arms = [(r, R0) for e, r, R0 in ((self.left_enabled, self.left_rot, self.init_left_arm_R),
                                         (self.right_enabled, self.right_rot, self.init_right_arm_R)) if e]
        loss += ARM_CAP_W * arm_limit_loss([r for r, _ in arms], [R0 for _, R0 in arms], self.arm_caps)

        return loss

    def all_losses(self):
        """Return the (keypoint, penetration, regularization, contact) losses."""
        smplh_params, _ = self.forward()
        output_verts, output_joints, output_cam_trans = run_body_model(self.smplh_model, smplh_params,
                                                                       self.cam, self.batch)
        joints_2d = project_joints_to_2d(output_joints, output_cam_trans, self.fl[0], self.img_w, self.img_h)
        output_verts = (output_verts + self.trans.unsqueeze(1))

        kp_loss = self._keypoint_loss_from_joints2d(joints_2d)
        pen_loss = self._penetration_loss_from_verts(output_verts)
        t_loss, self.n_tunnel = self.tunnel.loss((output_joints + self.trans.unsqueeze(1))[0],
                                                 self.tunnel_sides, self.move_coeff.clamp(0.95, 1.05))
        pen_loss = pen_loss + TUNNEL_W * t_loss
        pen_loss = pen_loss + ANCHOR_W * anchor_drift_loss(
            joints_2d, self.anchor_2d, output_joints + self.trans.unsqueeze(1), self.anchor_idx, self.fl[0],
            self.anchor_tol)
        reg_loss = self.regularization_loss()
        tact_loss = self._contact_loss_from_verts(output_verts)
        return kp_loss, pen_loss, reg_loss, tact_loss

    def _keypoint_loss_from_joints2d(self, joints_2d):
        """Fingers vs. WiLoR keypoints plus arms vs. their initial projection."""
        B = joints_2d.shape[0]
        loss = torch.zeros((), device=self.device, dtype=joints_2d.dtype)

        if self.left_enabled:
            corrected_left = joints_2d[:, self.left_idxs]
            corrected_left_arms = joints_2d[:, [16, 18, 20]]
            left_ref = self.left_ref
            if left_ref.dim() == 2:
                left_ref = left_ref.unsqueeze(0).expand(B, -1, -1)
            loss = loss + F.mse_loss(corrected_left, left_ref) + F.mse_loss(corrected_left_arms,
                                                                            self.init_left_arms)

        if self.right_enabled:
            corrected_right = joints_2d[:, self.right_idxs]
            corrected_right_arms = joints_2d[:, [17, 19, 21]]
            right_ref = self.right_ref
            if right_ref.dim() == 2:
                right_ref = right_ref.unsqueeze(0).expand(B, -1, -1)
            loss = loss + F.mse_loss(corrected_right, right_ref) + F.mse_loss(corrected_right_arms,
                                                                              self.init_right_arms)

        return loss

    def _penetration_loss_from_verts(self, output_verts):
        """Summed squared penetration of the palmar vertices."""
        loss = torch.tensor(0.0, device=self.device)

        if self.left_enabled and self.left_sdf_grid is not None:
            left_verts = output_verts[:, self.left_indices_tensor, :]
            s_left = self.sample_sdf_object_side(left_verts.reshape(-1, 3), "left")
            loss = loss + (F.relu(-s_left - PEN_MARGIN) ** 2).sum()

        if self.right_enabled and self.right_sdf_grid is not None:
            right_verts = output_verts[:, self.right_indices_tensor, :]
            s_right = self.sample_sdf_object_side(right_verts.reshape(-1, 3), "right")
            loss = loss + (F.relu(-s_right - PEN_MARGIN) ** 2).sum()

        return loss

    def _contact_loss_from_verts(self, output_verts):
        """Mean squared contact distance, averaged over hands."""
        B = output_verts.shape[0]
        s = self.move_coeff.clamp(0.95, 1.05)
        V_moved = s * self._obj_verts_t
        obj_V = V_moved.unsqueeze(0).expand(B, -1, -1)
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        n = 0
        for en, idx, ph, po in (
                (self.left_enabled, self.init_left_idx, self.pair_l_hand, self.pair_l_obj),
                (self.right_enabled, self.init_right_idx, self.pair_r_hand, self.pair_r_obj)):
            if not en:
                continue
            if ph.numel() >= 3:
                # paired object vert
                d = torch.norm(output_verts[:, ph, :] - V_moved[po].unsqueeze(0), dim=-1)
            elif idx.numel() > 0:
                # nearest object vert
                d2, _, _ = knn_points(output_verts[:, idx, :], obj_V, K=1)
                d = torch.sqrt(d2.squeeze(-1).clamp_min(1e-12))
            else:
                continue
            total = total + (F.relu(d - CONTACT_MARGIN) ** 2).mean()
            n += 1
        return total / max(n, 1)
