"""Elbow and wrist rotations fitted to the WiLoR keypoints."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from human_recon.constants import SMPL_MODEL_PATH, SMPLH_MODEL_PATH
from human_recon.constants import NUM_BETAS
from utils.projector_2d import project_joints_to_2d
from optimizer.common import run_body_model

WRIST_CONE_DEG = 30.0   # wrist rotation cone


class WristOptimizer(nn.Module):
    """SMPL-H wrist rotations fitted to the WiLoR 2D keypoints."""

    def __init__(self, *, left_kpts, right_kpts, smplh_params, cam, focal_length, batch,
                 left_flag: bool, right_flag: bool):
        super().__init__()
        self.device = torch.device('cuda')

        self.left_enabled  = bool(left_flag  and (left_kpts  is not None))
        self.right_enabled = bool(right_flag and (right_kpts is not None))
        if not (self.left_enabled or self.right_enabled):
            raise ValueError("Nothing to optimize: both sides disabled or missing refs.")

        self.smplh_model = smplx.SMPLHLayer(model_path=SMPLH_MODEL_PATH, num_betas=NUM_BETAS).to(self.device)
        self.smplh_params = {k: v.to(self.device) for k, v in smplh_params.items()}
        self.cam = cam.to(self.device)
        self.fl  = focal_length.to(self.device)
        self.batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        self.img_h, self.img_w = self.batch['img_size'][0].tolist()

        self.body_pose = self.smplh_params['body_pose'].detach().clone()

        self.register_buffer('left_ref',  torch.empty(0, 2, dtype=torch.float32, device=self.device))
        self.register_buffer('right_ref', torch.empty(0, 2, dtype=torch.float32, device=self.device))
        if self.left_enabled:
            self.left_ref  = torch.as_tensor(left_kpts[1:],  dtype=torch.float32, device=self.device)
        if self.right_enabled:
            self.right_ref = torch.as_tensor(right_kpts[1:], dtype=torch.float32, device=self.device)

        left_R  = matrix_to_axis_angle(self.body_pose[:, 19])
        right_R = matrix_to_axis_angle(self.body_pose[:, 20])
        self.left_rot  = nn.Parameter(left_R.clone(),  requires_grad=self.left_enabled)
        self.right_rot = nn.Parameter(right_R.clone(), requires_grad=self.right_enabled)

        self.left_idxs  = list(range(22, 22 + 15))
        self.right_idxs = list(range(22 + 15, 22 + 30))
        self.wrist_limit = float(np.radians(WRIST_CONE_DEG))
        self.register_buffer('init_left_R', axis_angle_to_matrix(left_R).detach().clone())
        self.register_buffer('init_right_R', axis_angle_to_matrix(right_R).detach().clone())

    @torch.no_grad()
    def project(self):
        """Clamp each wrist rotation to the WRIST_CONE_DEG cone around its start."""
        for rot, R0 in ((self.left_rot, self.init_left_R), (self.right_rot, self.init_right_R)):
            d = matrix_to_axis_angle(R0.transpose(-1, -2) @ axis_angle_to_matrix(rot))
            n = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            d = d * (n.clamp(max=self.wrist_limit) / n)
            rot.copy_(matrix_to_axis_angle(R0 @ axis_angle_to_matrix(d)))

    def forward(self):
        new_bp = self.body_pose.clone()
        if self.left_enabled:
            new_bp[:, 19] = axis_angle_to_matrix(self.left_rot)
        if self.right_enabled:
            new_bp[:, 20] = axis_angle_to_matrix(self.right_rot)
        self.smplh_params['body_pose'] = new_bp
        return self.smplh_params

    def loss(self):
        """Squared MSE of the projected finger joints to the WiLoR keypoints."""
        smplh_params = self.forward()
        _, output_joints, output_cam_trans = run_body_model(
            self.smplh_model, smplh_params, self.cam, self.batch
        )
        joints_2d = project_joints_to_2d(output_joints, output_cam_trans, self.fl[0], self.img_w, self.img_h)

        B = joints_2d.shape[0]
        loss = torch.zeros((), device=self.device, dtype=joints_2d.dtype)

        if self.left_enabled:
            corrected_left = joints_2d[:, self.left_idxs]
            left_ref = self.left_ref
            if left_ref.dim() == 2:
                left_ref = left_ref.unsqueeze(0).expand(B, -1, -1)
            loss = loss + F.mse_loss(corrected_left, left_ref) ** 2

        if self.right_enabled:
            corrected_right = joints_2d[:, self.right_idxs]
            right_ref = self.right_ref
            if right_ref.dim() == 2:
                right_ref = right_ref.unsqueeze(0).expand(B, -1, -1)
            loss = loss + F.mse_loss(corrected_right, right_ref) ** 2

        return loss


class WristPosOptimizer(nn.Module):
    """SMPL elbow rotations fitted to the WiLoR 2D wrist keypoints."""

    def __init__(self, *, smpl_params, cam, focal_length, batch,
                 left_wrist_ref=None, right_wrist_ref=None,
                 left_flag: bool = True, right_flag: bool = True):
        super().__init__()
        self.device = torch.device('cuda')
        self.left_enabled  = bool(left_flag)
        self.right_enabled = bool(right_flag)
        if not (self.left_enabled or self.right_enabled):
            raise ValueError("At least one of left_flag/right_flag must be True.")

        self.smpl_model = smplx.SMPLLayer(model_path=SMPL_MODEL_PATH, num_betas=NUM_BETAS).to(self.device)

        self.smpl_params = {k: v.to(self.device) for k, v in smpl_params.items()}
        self.base_body_pose = self.smpl_params['body_pose'].detach().clone().to(self.device)

        self.cam = cam.to(self.device)
        self.fl  = focal_length.to(self.device)
        self.batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        self.img_h, self.img_w = self.batch['img_size'][0].tolist()

        left_R  = matrix_to_axis_angle(self.base_body_pose[:, 17])
        right_R = matrix_to_axis_angle(self.base_body_pose[:, 18])
        self.Lelbow = nn.Parameter(left_R.clone(),  requires_grad=self.left_enabled)
        self.Relbow = nn.Parameter(right_R.clone(), requires_grad=self.right_enabled)

        if self.left_enabled:
            self.register_buffer('left_wrist_ref',
                torch.as_tensor(left_wrist_ref[0], dtype=torch.float32, device=self.device))
        else:
            self.register_buffer('left_wrist_ref', torch.empty(0, device=self.device))
        if self.right_enabled:
            self.register_buffer('right_wrist_ref',
                torch.as_tensor(right_wrist_ref[0], dtype=torch.float32, device=self.device))
        else:
            self.register_buffer('right_wrist_ref', torch.empty(0, device=self.device))

    def forward(self):
        new_bp = self.base_body_pose.clone()
        if self.left_enabled:
            new_bp[:, 17] = axis_angle_to_matrix(self.Lelbow)
        if self.right_enabled:
            new_bp[:, 18] = axis_angle_to_matrix(self.Relbow)
        self.smpl_params['body_pose'] = new_bp

        _, output_joints, output_cam_trans = run_body_model(
            self.smpl_model, self.smpl_params, self.cam, self.batch
        )
        joints_2d = project_joints_to_2d(output_joints, output_cam_trans, self.fl[0], self.img_w, self.img_h)
        return joints_2d[0], self.smpl_params

    def loss(self):
        """Pixel distance of the projected wrists to the WiLoR keypoints."""
        joints, _ = self.forward()
        loss = torch.zeros((), device=self.device, dtype=torch.float32)

        if self.left_enabled:
            loss = loss + (joints[20] - self.left_wrist_ref).norm()
        if self.right_enabled:
            loss = loss + (joints[21] - self.right_wrist_ref).norm()
        return loss
