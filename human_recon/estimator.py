"""Human initialization from the SAM 3D Body fit."""
import numpy as np
import torch
import smplx
from smplx.lbs import batch_rodrigues

from human_recon.constants import SMPL_MODEL_PATH, SMPLH_MODEL_PATH, NUM_BETAS
from optimizer.common import run_body_model as _run_body_model

BODY_PARAMS = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "betas", "trans")


class HumanEstimator:
    """SAM 3D Body fit of the current sample and the SMPL/SMPL-H layers."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.smpl_model = smplx.SMPLLayer(model_path=SMPL_MODEL_PATH, num_betas=NUM_BETAS).to(self.device)
        self.smplh_model = smplx.SMPLHLayer(model_path=SMPLH_MODEL_PATH, num_betas=NUM_BETAS).to(self.device)
        self._cur = None            # current sample

    def set_sample(self, params, bbox_xyxy, K):
        """Set the current sample: body fit (axis-angle), person box and K."""
        self._cur = {k: np.asarray(params[k], dtype=np.float32) for k in BODY_PARAMS}
        self._cur["bbox"] = np.asarray(bbox_xyxy, dtype=np.float32)
        self._cur["cam_int"] = np.asarray(K, dtype=np.float32)

    def _sample(self):
        if self._cur is None:
            raise RuntimeError("HumanEstimator.set_sample() must be called first")
        return self._cur

    def person_box(self):
        """Person box as (center, scale), each (1, 2); scale is box size / 200."""
        x1, y1, x2, y2 = [float(v) for v in self._sample()["bbox"]]
        bbox_center = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], dtype=np.float32)
        bbox_scale = np.array([[(x2 - x1) / 200.0, (y2 - y1) / 200.0]], dtype=np.float32)
        return bbox_center, bbox_scale

    def cam_intrinsics(self):
        """HumanFoV intrinsic matrix K (3, 3)."""
        return self._sample()["cam_int"].astype(np.float32)

    def body_batch(self, img_cv2, bbox_center, bbox_scale, cam_int):
        """Camera and box batch of run_body_model."""
        H, W = img_cv2.shape[:2]
        box_wh = bbox_scale[0] * 200.0
        box_size = float(max(box_wh[0], box_wh[1]))
        batch = {
            "img_size": torch.tensor([[H, W]], dtype=torch.float32, device=self.device),
            "box_size": torch.tensor([box_size], dtype=torch.float32, device=self.device),
            "box_center": torch.tensor(bbox_center, dtype=torch.float32, device=self.device),
            "cam_int": torch.tensor(cam_int, dtype=torch.float32, device=self.device)[None]
            if np.ndim(cam_int) == 2 else torch.tensor(cam_int, dtype=torch.float32, device=self.device),
        }
        return batch

    def body_forward(self, batch):
        """Fit as (SMPL rotation matrices, weak-perspective camera, focal length)."""
        c = self._sample()
        dev = self.device
        go = torch.tensor(c["global_orient"], dtype=torch.float32, device=dev).reshape(1, 3)
        bp = torch.tensor(c["body_pose"], dtype=torch.float32, device=dev).reshape(21, 3)
        betas = torch.tensor(c["betas"], dtype=torch.float32, device=dev).reshape(1, NUM_BETAS)
        trans = torch.tensor(c["trans"], dtype=torch.float32, device=dev).reshape(3)

        global_orient = batch_rodrigues(go).reshape(1, 1, 3, 3)
        body21 = batch_rodrigues(bp).reshape(1, 21, 3, 3)
        # 21 SMPL-H joints + 2 identity hand roots
        eye2 = torch.eye(3, device=dev).reshape(1, 1, 3, 3).expand(1, 2, 3, 3)
        body_pose = torch.cat([body21, eye2], dim=1)
        out_smpl_params = {"global_orient": global_orient, "body_pose": body_pose, "betas": betas}

        # inverse of convert_to_full_img_cam
        f = float(batch["cam_int"][0, 0, 0])
        box = float(batch["box_size"][0])
        bcx, bcy = float(batch["box_center"][0, 0]), float(batch["box_center"][0, 1])
        H, W = float(batch["img_size"][0, 0]), float(batch["img_size"][0, 1])
        Tx, Ty, Tz = float(trans[0]), float(trans[1]), float(trans[2])
        s = 2.0 * f / (box * Tz)
        cx = 2.0 * (bcx - W / 2.0) / (s * box)
        cy = 2.0 * (bcy - H / 2.0) / (s * box)
        out_cam = torch.tensor([[s, Tx - cx, Ty - cy]], dtype=torch.float32, device=dev)
        focal_length_ = torch.tensor([f], dtype=torch.float32, device=dev)
        return out_smpl_params, out_cam, focal_length_

    def fitted_hand_rotmats(self):
        """Fitted finger rotation matrices (left, right), each (1, 15, 3, 3)."""
        c = self._sample()
        out = []
        for k in ("left_hand_pose", "right_hand_pose"):
            aa = torch.tensor(c[k], dtype=torch.float32, device=self.device).reshape(15, 3)
            out.append(batch_rodrigues(aa).reshape(1, 15, 3, 3))
        return out[0], out[1]

    def run_body_model(self, body_model, params, pred_cam, batch):
        """Pose a body layer. Returns (vertices, joints, cam_trans)."""
        return _run_body_model(body_model, params, pred_cam, batch)
