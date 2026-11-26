"""Box and back-projection helpers."""
import numpy as np
import cv2
import torch
from pytorch3d.renderer.cameras import CamerasBase


class BboxCalc:
    """Pixel bounding box of projected vertices."""

    def __init__(self, cameras: CamerasBase):
        self.cameras = cameras

    @torch.no_grad()
    def get_bbox_from_verts(self, verts: torch.Tensor) -> torch.Tensor:
        """(x_min, y_min, w, h) in pixels of (V, 3) vertices."""
        valid = verts[:, 2] > 1e-6
        if not bool(valid.any()):
            return torch.tensor([0., 0., 0., 0.], device=verts.device)

        verts = verts[valid]

        # image_size is [(H, W)]
        screen = self.cameras.transform_points_screen(
            verts[None], image_size=self.cameras.image_size
        )[0]

        u = screen[:, 0]
        v = screen[:, 1]

        x_min = u.min()
        y_min = v.min()
        x_max = u.max()
        y_max = v.max()

        w = x_max - x_min
        h = y_max - y_min
        return torch.stack([x_min, y_min, w, h])


def obj_bbox_center(mask: np.ndarray):
    """Center x, y, width and height (px) of the nonzero mask's box."""
    binary_mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No non-zero regions found in mask.")

    x, y, w, h = cv2.boundingRect(np.vstack(contours))

    center_x = x + w / 2.0
    center_y = y + h / 2.0

    return center_x, center_y, w, h


def image_to_camera_coords(x, y, z, img_w, img_h, focal_length):
    """Back-project pixel (x, y) at depth z to camera coordinates."""
    focal_length = focal_length[0].item()
    cx = img_w / 2.0
    cy = img_h / 2.0
    X = (x - cx) * z / focal_length
    Y = (y - cy) * z / focal_length
    X = X.detach().cpu().item()
    Y = Y.detach().cpu().item()
    return np.array([X, Y, z], dtype=np.float32)
