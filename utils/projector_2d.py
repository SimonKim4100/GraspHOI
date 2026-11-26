"""Pinhole projection and square crop boxes."""
import numpy as np
import torch


def project_joints_to_2d(joints_3d, cam_trans, focal_length, img_w, img_h):
    """Project (B, N, 3) points plus cam_trans (B, 3) to (B, N, 2) pixels."""
    if torch.is_tensor(joints_3d):
        device = joints_3d.device
        dtype  = joints_3d.dtype
    elif torch.is_tensor(cam_trans):
        device = cam_trans.device
        dtype  = cam_trans.dtype
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype  = torch.float32

    joints_3d = (joints_3d if torch.is_tensor(joints_3d)
                 else torch.as_tensor(joints_3d, dtype=dtype, device=device))
    cam_trans = (cam_trans if torch.is_tensor(cam_trans)
                 else torch.as_tensor(cam_trans, dtype=joints_3d.dtype, device=device))
    focal_length = (focal_length if torch.is_tensor(focal_length)
                    else torch.as_tensor(focal_length, dtype=joints_3d.dtype, device=device))

    cam_trans = cam_trans.to(device=joints_3d.device, dtype=joints_3d.dtype)
    focal_length = focal_length.to(device=joints_3d.device, dtype=joints_3d.dtype)

    joints_cam = joints_3d + cam_trans.unsqueeze(1)  # (B, N, 3)
    x, y, z = joints_cam[..., 0], joints_cam[..., 1], joints_cam[..., 2].clamp(min=1e-6)

    fx = fy = focal_length
    cx = (img_w - 1) / 2.0
    cy = (img_h - 1) / 2.0

    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return torch.stack([u, v], dim=-1)


def make_square_bbox(bbox, img_w, img_h, scale=1.0):
    """Square xyxy box (int pixels) of side scale x the longer bbox side."""
    if torch.is_tensor(bbox):
        bbox = bbox.detach().cpu().numpy()
    x1, y1, x2, y2 = map(float, np.asarray(bbox).tolist())

    if torch.is_tensor(img_w): img_w = int(img_w.detach().cpu().item())
    else:                      img_w = int(img_w)
    if torch.is_tensor(img_h): img_h = int(img_h.detach().cpu().item())
    else:                      img_h = int(img_h)

    w = x2 - x1
    h = y2 - y1
    size = max(w, h) * float(scale)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    new_x1 = max(0, int(cx - size / 2.0))
    new_y1 = max(0, int(cy - size / 2.0))
    new_x2 = min(img_w - 1, int(cx + size / 2.0))
    new_y2 = min(img_h - 1, int(cy + size / 2.0))

    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
