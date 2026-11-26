"""WiLoR hand reconstruction on detected hand boxes."""
import torch
import numpy as np

from find_mano.wilor.models import load_wilor, download_checkpoint, DEFAULT_CHECKPOINT, DEFAULT_CONFIG
from find_mano.wilor.utils import recursive_to
from find_mano.wilor.datasets.vitdet_dataset import ViTDetDataset
from find_mano.wilor.utils.renderer import cam_crop_to_full


class Wilor:
    """WiLoR on given hand boxes."""

    def __init__(self, batch_size=16, device=None):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        download_checkpoint()
        self.model, self.model_cfg = load_wilor(checkpoint_path=DEFAULT_CHECKPOINT, cfg_path=DEFAULT_CONFIG)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    def run(self, img_cv2, boxes, is_right, left_flag, right_flag, rescale_factor=2.0):
        """Run WiLoR on the hand boxes, left first. Returns (mano_params, kpts)."""
        boxes = np.stack(boxes)
        right = np.stack(is_right)

        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                                 num_workers=0)

        batch = next(iter(dataloader))  # at most two hands
        batch = recursive_to(batch, self.device)

        with torch.no_grad():
            out = self.model(batch)

        # index 0: left hand if detected
        hand_pose = out['pred_mano_params']['hand_pose']  # (B, 15, 3, 3)
        left_global_orient = out['pred_mano_params']['global_orient'][0]
        left_hand_pose = hand_pose[0]
        right_global_orient = out['pred_mano_params']['global_orient'][0]
        right_hand_pose = hand_pose[0]
        if left_flag and right_flag:
            right_global_orient = out['pred_mano_params']['global_orient'][1]
            right_hand_pose = hand_pose[1]

        # from WiLoR demo.py (https://github.com/rolpotamias/WiLoR), CC BY-NC-ND 4.0, see find_mano/wilor/LICENSE
        # crop camera -> full-image translation
        multiplier    = (2*batch['right']-1)
        pred_cam      = out['pred_cam']
        pred_cam[:,1] = multiplier*pred_cam[:,1]
        box_center    = batch["box_center"].float()
        box_size      = batch["box_size"].float()
        img_size      = batch["img_size"].float()            # (B,2) [W,H]
        scaled_focal_length = (self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE
                               * img_size.max())
        pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                               scaled_focal_length).detach().cpu().numpy()

        B = batch['img'].shape[0]
        all_kpts = []

        # wrist + 3 joints per finger, in SMPL-H order
        keep = [0, 5,6,7,  9,10,11,  17,18,19,  13,14,15,  1,2,3]

        for n in range(B):
            joints3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()
            right_n = int(batch['right'][n].item())

            joints = joints3d.copy()
            joints[:,0] = (2*right_n-1)*joints[:,0]    # left hands are predicted mirrored
            joints16 = joints[keep]
            cam_t = pred_cam_t_full[n]
            kpts_2d = project_full_img(joints16, cam_t, scaled_focal_length, img_size[n])
            all_kpts.append(kpts_2d)

        # mirror the left-hand rotations back
        flip = torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]], device=left_hand_pose.device,
                            dtype=left_hand_pose.dtype)
        left_hand_pose     = flip @ left_hand_pose @ flip
        left_global_orient = flip @ left_global_orient @ flip

        result = {
            'left_global_orient':  left_global_orient.unsqueeze(0),
            'right_global_orient': right_global_orient.unsqueeze(0),
            'left_hand_pose':      left_hand_pose.unsqueeze(0),
            'right_hand_pose':     right_hand_pose.unsqueeze(0),
        }
        return result, all_kpts


# from WiLoR demo.py (https://github.com/rolpotamias/WiLoR), CC BY-NC-ND 4.0, see find_mano/wilor/LICENSE
def project_full_img(points, cam_trans, focal_length, img_res):
    """Project points + cam_trans to pixels; img_res = [W, H]."""
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3)
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:]

    V_2d = (K @ points.T).T
    return V_2d[..., :-1]
