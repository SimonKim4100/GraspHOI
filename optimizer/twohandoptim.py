"""Coarse two-arm stage of the joint refinement, fingers and object fixed."""
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
    run_body_model, select_contact_vertices, first_or_empty, reg_distance,
    TunnelGuard, FINGER_CHAINS, ARM_JOINTS, anchor_drift_loss, arm_limit_loss,
    TUNNEL_W, ANCHOR_W, ANCHOR_TOL_HAND, ANCHOR_TOL_ARM, ARM_CAPS_DEG, ARM_CAP_W,
    LEFT_HAND_VERTS, RIGHT_HAND_VERTS,
)

PEN_MARGIN = 0.002      # metres
CONTACT_MARGIN = 0.0    # metres


class TwoHandOptimizer(nn.Module):
    """Shoulder, elbow and wrist rotations of both arms."""

    def __init__(self, *, smplh_params, obj_verts, obj_faces, cam, focal_length, batch, smplh_verts,
                 left_kpts, right_kpts, amodal_mask,
                 contact_pairs=None,
                 reg_w_arm: float = 0.5):
        super().__init__()
        self.device = torch.device('cuda')

        if left_kpts is None or right_kpts is None:
            raise ValueError("This class assumes BOTH hands are active: provide left_kpts and right_kpts.")

        self.obj_verts = obj_verts
        self.obj_faces = obj_faces
        self.smplh_verts = torch.as_tensor(smplh_verts, dtype=torch.float32, device=self.device).contiguous()
        self.smplh_model = smplx.SMPLHLayer(model_path=SMPLH_MODEL_PATH, num_betas=NUM_BETAS).to(self.device)
        self.smplh_params = {k: v.to(self.device) for k, v in smplh_params.items()}
        self.cam = cam.to(self.device)
        self.fl = focal_length.to(self.device)
        self.batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        self.img_h, self.img_w = self.batch['img_size'][0].tolist()
        self.reg_w_arm = float(reg_w_arm)

        self.left_verts_idx = LEFT_HAND_VERTS
        self.right_verts_idx = RIGHT_HAND_VERTS

        output_verts, output_joints, output_cam_trans = run_body_model(self.smplh_model, self.smplh_params,
                                                                       self.cam, self.batch)
        self.trans = output_cam_trans
        init_joints_2d = project_joints_to_2d(
            output_joints, output_cam_trans, self.fl[0], self.img_w, self.img_h
        )

        with torch.no_grad():
            obj_faces_np = obj_faces.detach().long().cpu().numpy()
        self.obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces_np, process=False)
        self.tunnel = TunnelGuard(self.obj_mesh.vertices, self.obj_mesh.faces)
        self.n_tunnel = 0

        # amodal silhouette (H, W)
        amodal_mask = np.asarray(amodal_mask)
        if amodal_mask.dtype != np.uint8:
            raise TypeError(f"amodal_mask must be the uint8 (0 / 255) S_a, got {amodal_mask.dtype}")
        self.object_mask = torch.from_numpy(amodal_mask >= 128).to(self.device)

        self.body_pose = self.smplh_params['body_pose'].detach().clone()

        # WiLoR 2D keypoints without the wrist
        self.register_buffer('left_ref',  torch.as_tensor(left_kpts[1:],  dtype=torch.float32,
                                                          device=self.device))
        self.register_buffer('right_ref', torch.as_tensor(right_kpts[1:], dtype=torch.float32,
                                                          device=self.device))

        # SMPL-H finger joints
        self.left_idxs  = list(range(22, 22 + 15))
        self.right_idxs = list(range(22 + 15, 22 + 30))

        sdf_grid, origin_xyz, voxel = self.mesh_to_sdf_grid(voxel_size=0.005, pad=0.02)
        self.register_buffer("obj_sdf_grid", sdf_grid)
        self.register_buffer("obj_sdf_origin", origin_xyz)
        self.obj_sdf_voxel = float(voxel)

        # shoulder, elbow, wrist
        left_R  = matrix_to_axis_angle(self.body_pose[:, [15, 17, 19]])
        right_R = matrix_to_axis_angle(self.body_pose[:, [16, 18, 20]])
        self.left_rot  = nn.Parameter(left_R.clone(),  requires_grad=True)
        self.right_rot = nn.Parameter(right_R.clone(), requires_grad=True)

        self.register_buffer('init_left_arm_R',  self.body_pose[:, [15, 17, 19]].detach().clone())
        self.register_buffer('init_right_arm_R', self.body_pose[:, [16, 18, 20]].detach().clone())

        # arm anchors of the keypoint loss
        self.init_left_arms = init_joints_2d[:, [16, 18, 20]]
        self.init_right_arms = init_joints_2d[:, [17, 19, 21]]
        # 2D trust region
        hand_j = sorted({j for s in ["left", "right"] for c in FINGER_CHAINS[s] for j in c})
        arm_j = sorted({j for s in ["left", "right"] for j in ARM_JOINTS[s]})
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
            left_enabled=True, right_enabled=True
        )
        self.init_left_idx  = first_or_empty(sel.get("left_idx_in", []),  self.device)
        self.init_right_idx = first_or_empty(sel.get("right_idx_in", []), self.device)

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

        # fallback: nearest hand verts
        sv = self.smplh_verts[0].detach().cpu().numpy()
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

        self.init_left_idx = _ensure_candidates(self.init_left_idx, self.left_verts_idx, "twohand-L")
        self.init_right_idx = _ensure_candidates(self.init_right_idx, self.right_verts_idx, "twohand-R")

    @staticmethod
    def sample_sdf_world(points_world, sdf_grid, origin_xyz, voxel_size, align_corners=True):
        """Sample an SDF grid (1, 1, Nz, Ny, Nx) at world points (N, 3)."""
        device = sdf_grid.device
        dtype  = sdf_grid.dtype
        points_world = points_world.to(device=device, dtype=dtype)
        origin_xyz   = origin_xyz.to(device=device, dtype=dtype)

        D, H, W = sdf_grid.shape[-3:]
        g = (points_world - origin_xyz) / voxel_size
        gx, gy, gz = g.unbind(-1)
        nx = 2.0 * (gx / (W - 1)) - 1.0
        ny = 2.0 * (gy / (H - 1)) - 1.0
        nz = 2.0 * (gz / (D - 1)) - 1.0
        grid = torch.stack([nx, ny, nz], dim=-1).view(1, -1, 1, 1, 3)

        vals = F.grid_sample(
            sdf_grid, grid.to(dtype).to(device),
            mode='bilinear', padding_mode='border', align_corners=align_corners
        )
        return vals.view(-1)

    def mesh_to_sdf_grid(self, voxel_size=0.005, pad=0.02, chunk=4_000_000, max_voxels=64_000_000):
        """Object SDF over the padded object bbox."""
        mn, mx = self.obj_mesh.bounds
        mn = mn.astype(np.float32, copy=True)
        mx = mx.astype(np.float32, copy=True)
        mn -= pad; mx += pad

        # bound the grid size
        span = (mx - mn).astype(np.float64)
        total = np.prod(np.ceil(span / voxel_size) + 1)
        if total > max_voxels:
            voxel_size = float(voxel_size * (total / max_voxels) ** (1.0 / 3.0))
            print(f"[twohand] WARNING: object span {np.round(span, 2)} m too large for the SDF grid "
                  f"-> voxel coarsened to {voxel_size:.4f} m", flush=True)

        spans = (mx - mn) / voxel_size
        Nx, Ny, Nz = (np.ceil(spans) + 1).astype(int)
        xs = np.linspace(mn[0], mx[0], Nx, dtype=np.float32)
        ys = np.linspace(mn[1], mx[1], Ny, dtype=np.float32)
        zs = np.linspace(mn[2], mx[2], Nz, dtype=np.float32)

        Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        # negative inside the object
        sdf = np.empty((pts.shape[0],), dtype=np.float32)
        import igl
        obj_faces_np = self.obj_faces.detach().cpu().numpy().astype(np.int32, copy=False)
        for a in range(0, pts.shape[0], chunk):
            b = min(a + chunk, pts.shape[0])
            s, _, _, _ = igl.signed_distance(
                pts[a:b], self.obj_verts, obj_faces_np, igl.SIGNED_DISTANCE_TYPE_WINDING_NUMBER
            )
            sdf[a:b] = np.asarray(s, np.float32).ravel()
        sdf = sdf.reshape(Nz, Ny, Nx)

        sdf_grid = torch.from_numpy(sdf).unsqueeze(0).unsqueeze(0).to(self.device)
        origin_xyz = torch.tensor(mn.astype(np.float32), device=self.device)
        return sdf_grid, origin_xyz, float(voxel_size)

    def forward(self):
        new_bp = self.body_pose.clone()

        L_arm = axis_angle_to_matrix(self.left_rot)
        new_bp[:, 15] = L_arm[:, 0]
        new_bp[:, 17] = L_arm[:, 1]
        new_bp[:, 19] = L_arm[:, 2]

        R_arm = axis_angle_to_matrix(self.right_rot)
        new_bp[:, 16] = R_arm[:, 0]
        new_bp[:, 18] = R_arm[:, 1]
        new_bp[:, 20] = R_arm[:, 2]

        self.smplh_params['body_pose'] = new_bp
        return self.smplh_params

    def regularization_loss(self):
        """Distance to the initial arm rotations plus the rotation caps."""
        loss = torch.zeros((), device=self.device, dtype=self.body_pose.dtype)

        R_arm_L  = axis_angle_to_matrix(self.left_rot)
        R0_arm_L = self.init_left_arm_R
        base_L   = reg_distance(R_arm_L, R0_arm_L).mean()
        loss    += self.reg_w_arm * base_L

        R_arm_R  = axis_angle_to_matrix(self.right_rot)
        R0_arm_R = self.init_right_arm_R
        base_R   = reg_distance(R_arm_R, R0_arm_R).mean()
        loss    += self.reg_w_arm * base_R

        loss += ARM_CAP_W * arm_limit_loss([self.left_rot, self.right_rot],
                                           [self.init_left_arm_R, self.init_right_arm_R], self.arm_caps)

        return loss

    def all_losses(self):
        """Return the (keypoint, penetration, contact, regularization) losses."""
        smplh_params = self.forward()
        output_verts, output_joints, output_cam_trans = run_body_model(self.smplh_model, smplh_params,
                                                                       self.cam, self.batch)
        joints_2d = project_joints_to_2d(output_joints, output_cam_trans, self.fl[0], self.img_w, self.img_h)
        output_verts = (output_verts + self.trans.unsqueeze(1))

        kp_loss = self._keypoint_loss_from_joints2d(joints_2d)
        pen_loss = self._penetration_loss_from_verts(output_verts)
        t_loss, self.n_tunnel = self.tunnel.loss((output_joints + self.trans.unsqueeze(1))[0],
                                                 ["left", "right"])
        pen_loss = pen_loss + TUNNEL_W * t_loss
        pen_loss = pen_loss + ANCHOR_W * anchor_drift_loss(
            joints_2d, self.anchor_2d, output_joints + self.trans.unsqueeze(1), self.anchor_idx, self.fl[0],
            self.anchor_tol)
        tact_loss = self._contact_loss_from_verts(output_verts)
        reg_loss = self.regularization_loss()
        return kp_loss, pen_loss, tact_loss, reg_loss

    def _keypoint_loss_from_joints2d(self, joints_2d):
        """Fingers vs. WiLoR keypoints plus arms vs. their initial projection."""
        B = joints_2d.shape[0]
        loss = torch.zeros((), device=self.device, dtype=joints_2d.dtype)

        corrected_left = joints_2d[:, self.left_idxs]
        corrected_left_arms = joints_2d[:, [16, 18, 20]]
        left_ref = self.left_ref
        if left_ref.dim() == 2:
            left_ref = left_ref.unsqueeze(0).expand(B, -1, -1)
        loss = loss + F.mse_loss(corrected_left, left_ref) + F.mse_loss(corrected_left_arms,
                                                                        self.init_left_arms)

        corrected_right = joints_2d[:, self.right_idxs]
        corrected_right_arms = joints_2d[:, [17, 19, 21]]
        right_ref = self.right_ref
        if right_ref.dim() == 2:
            right_ref = right_ref.unsqueeze(0).expand(B, -1, -1)
        loss = loss + F.mse_loss(corrected_right, right_ref) + F.mse_loss(corrected_right_arms,
                                                                          self.init_right_arms)

        return loss

    def _penetration_loss_from_verts(self, output_verts):
        """Summed squared penetration of all body vertices."""
        s = self.sample_sdf_world(output_verts.reshape(-1, 3), self.obj_sdf_grid, self.obj_sdf_origin,
                                  self.obj_sdf_voxel)
        pen = F.relu(-s - PEN_MARGIN)
        return (pen ** 2).sum()

    def _contact_loss_from_verts(self, output_verts):
        """Mean unsquared contact distance, averaged over hands."""
        B = output_verts.shape[0]
        obj_V = self._obj_verts_t.unsqueeze(0).expand(B, -1, -1)
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        n = 0
        for idx, ph, po in ((self.init_left_idx, self.pair_l_hand, self.pair_l_obj),
                            (self.init_right_idx, self.pair_r_hand, self.pair_r_obj)):
            if ph.numel() >= 3:
                # paired object vert
                d = torch.norm(output_verts[:, ph, :] - self._obj_verts_t[po].unsqueeze(0), dim=-1)
            elif idx.numel() > 0:
                # nearest object vert
                d2, _, _ = knn_points(output_verts[:, idx, :], obj_V, K=1)
                d = torch.sqrt(d2).squeeze(-1)
            else:
                continue
            total = total + F.relu(d - CONTACT_MARGIN).mean()
            n += 1
        return total / max(n, 1)
