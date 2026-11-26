"""Silhouette alignment of the reconstructed object (L_obj)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import kornia.filters as KF
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    BlendParams,
    MeshRenderer,
    SoftSilhouetteShader,
    TexturesVertex,
    PerspectiveCameras,
)
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from utils.box_utils import BboxCalc, obj_bbox_center, image_to_camera_coords
from optimizer.common import LEFT_FINGERTIPS, RIGHT_FINGERTIPS, get_2d_bbox_from_verts

# soft silhouette rasterization
SOFT_SIGMA = 1e-4
SOFT_BLUR = float(np.log(1.0 / 1e-4 - 1.0) * SOFT_SIGMA)
SOFT_FACES_PER_PIXEL = 25


class ObjectOptimizer(nn.Module):
    """Object pose and scale fitted to the amodal silhouette."""

    def __init__(self, *, og_obj_verts, obj_faces, smplh_verts, R_mat, img_w, img_h,
                 focal_length, hands_detected, amodal_mask, init_from_pose=False, translation=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.og_obj_verts = torch.as_tensor(og_obj_verts, dtype=torch.float32,
                                            device=self.device).clone().detach()
        self.obj_faces = torch.as_tensor(obj_faces, dtype=torch.int64, device=self.device).clone().detach()
        smplh_verts_t = torch.tensor(smplh_verts, dtype=torch.float32, device=self.device)
        self.R_mat = torch.tensor(R_mat, dtype=torch.float32, device=self.device)
        self.img_w = torch.tensor(img_w, dtype=torch.float32, device=self.device)
        self.img_h = torch.tensor(img_h, dtype=torch.float32, device=self.device)
        self.focal_length = torch.tensor(focal_length, dtype=torch.float32, device=self.device)
        self.init_from_pose = bool(init_from_pose)

        fx, fy = self.focal_length
        W, H = int(self.img_w), int(self.img_h)
        R = torch.eye(3).unsqueeze(0).to(self.device)
        T = torch.zeros(1, 3).to(self.device)

        verts_np = smplh_verts_t[0].detach().cpu().numpy()
        left_tips = np.mean(verts_np[LEFT_FINGERTIPS], axis=0)
        right_tips = np.mean(verts_np[RIGHT_FINGERTIPS], axis=0)

        object_mask = np.asarray(amodal_mask)
        center_x, center_y, img_obj_w, img_obj_h = obj_bbox_center(object_mask)

        # square fit window around the mask
        R_CROP = 384
        S = float(np.clip(max(img_obj_w, img_obj_h) * 1.8, 48.0, min(W, H)))
        x0 = float(np.clip(center_x - S / 2.0, 0.0, max(W - S, 0.0)))
        y0 = float(np.clip(center_y - S / 2.0, 0.0, max(H - S, 0.0)))
        x0, y0, S = int(round(x0)), int(round(y0)), int(round(S))
        zoom = R_CROP / float(S)
        self.crop = {"x0": x0, "y0": y0, "S": S, "R": R_CROP}
        self.crop_fl = (float(fx) * zoom, float(fy) * zoom)
        self.crop_pp = ((float(W) / 2.0 - x0) * zoom, (float(H) / 2.0 - y0) * zoom)

        self.cameras = PerspectiveCameras(
            focal_length=[self.crop_fl],
            principal_point=[self.crop_pp],
            image_size=[(R_CROP, R_CROP)],
            in_ndc=False, R=R, T=T, device=self.device
        )
        bbox_object = BboxCalc(self.cameras)

        # target mask and box in crop pixels
        mcrop = object_mask[y0:y0 + S, x0:x0 + S]
        mcrop = cv2.resize(mcrop, (R_CROP, R_CROP), interpolation=cv2.INTER_NEAREST)
        object_mask_tensor = torch.tensor(mcrop, dtype=torch.float32, device=self.device)
        self.bbox_img_size = torch.tensor([img_obj_w * zoom, img_obj_h * zoom],
                                          dtype=torch.float32, device=self.device)
        self.bbox_center = torch.tensor([(center_x - x0) * zoom, (center_y - y0) * zoom],
                                        dtype=torch.float32, device=self.device)

        # no dropped faces
        self.raster_settings = RasterizationSettings(
            image_size=(R_CROP, R_CROP), blur_radius=SOFT_BLUR,
            faces_per_pixel=SOFT_FACES_PER_PIXEL,
            bin_size=None, max_faces_per_bin=max(int(self.obj_faces.shape[0]), 10000))
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=SOFT_SIGMA, gamma=1e-3))
        )

        use_pose = self.init_from_pose and translation is not None
        if use_pose:
            # implausible pose -> mask-centered init
            t0 = np.asarray(translation, dtype=np.float32)
            verts0 = (self.R_mat @ self.og_obj_verts.T).T + torch.tensor(t0, device=self.device)
            bb0 = bbox_object.get_bbox_from_verts(verts0)
            w0, h0 = float(bb0[2]), float(bb0[3])
            corr = float(np.sqrt((img_obj_w * zoom / max(w0, 1e-6)) *
                                 (img_obj_h * zoom / max(h0, 1e-6))))
            if float(t0[2]) <= 0.05 or not (0.25 <= corr <= 4.0):
                print(f"[objoptim] implausible starting pose (z={float(t0[2]):.3f}, "
                      f"scale correction {corr:.2f}) -> mask-centered init", flush=True)
                use_pose = False

        if use_pose:
            # start from the given pose
            self.obj_translation = torch.tensor(translation, dtype=torch.float32, device=self.device)
            c_rot = self.R_mat @ self.og_obj_verts.mean(0)
            self.obj_translation = self.obj_translation + (1.0 - corr) * c_rot
            self.rescale = corr
            self.obj_verts = (self.R_mat @ self.og_obj_verts.T).T * corr + self.obj_translation
        else:
            # mask-centered init at the fingertip depth
            if hands_detected[0] and hands_detected[1]:
                hand_z = float((left_tips[2] + right_tips[2]) / 2.0)
            elif hands_detected[0]:
                hand_z = float(left_tips[2])
            else:
                hand_z = float(right_tips[2])

            obj_translation_np = image_to_camera_coords(
                center_x, center_y, hand_z, self.img_w, self.img_h, self.focal_length
            )
            self.obj_translation = torch.tensor(obj_translation_np, dtype=torch.float32, device=self.device)
            reverse_R = trans_to_rot(self.obj_translation)
            self.R_mat = reverse_R @ self.R_mat

            self.obj_verts = (self.R_mat @ self.og_obj_verts.T).T + self.obj_translation
            _, _, _, h = bbox_object.get_bbox_from_verts(self.obj_verts)   # crop px
            mesh_obj_h = h.clamp_min(1e-6)
            img_obj_h = torch.tensor(img_obj_h * zoom, dtype=torch.float32, device=self.device)
            self.rescale = float(img_obj_h / mesh_obj_h)
            self.obj_verts = (self.R_mat @ self.og_obj_verts.T).T * self.rescale + self.obj_translation

        # target mask and contour distance field
        gt_mask = object_mask_tensor.float()
        if gt_mask.max() > 1:
            gt_mask = gt_mask / 255.0
        self.register_buffer('gt_mask_norm', gt_mask)
        _dt2, _edge = _precompute_contour_dt2(gt_mask)
        self.register_buffer('_gt_dt2', _dt2)
        self.register_buffer('_gt_edge', _edge)

        # OpenCV camera axes -> PyTorch3D
        self.register_buffer('_flip_matrix', torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            device=self.device
        ))
        self._white_tex = TexturesVertex(
            verts_features=torch.ones(1, self.og_obj_verts.shape[0], 3, device=self.device)
        )

        initial_rotvec = matrix_to_axis_angle(self.R_mat.unsqueeze(0))[0]
        self.position = nn.Parameter(self.obj_translation.clone())
        self.orientation = nn.Parameter(initial_rotvec.clone())
        self.scale = nn.Parameter(torch.tensor(self.rescale, dtype=torch.float32, device=self.device))

    def forward(self):
        R = axis_angle_to_matrix(self.orientation.unsqueeze(0)).squeeze(0)
        return (R @ self.og_obj_verts.T).T * self.scale + self.position

    def crop_full_map(self, m, nearest=True):
        """Crop and resize a full-frame map into the fit window."""
        c = self.crop
        x0, y0, S, Rc = c["x0"], c["y0"], c["S"], c["R"]
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        if torch.is_tensor(m):
            sub = m.detach().cpu().numpy()[y0:y0 + S, x0:x0 + S]
            return torch.tensor(cv2.resize(sub, (Rc, Rc), interpolation=interp),
                                dtype=m.dtype, device=m.device)
        sub = np.asarray(m)[y0:y0 + S, x0:x0 + S]
        return cv2.resize(sub, (Rc, Rc), interpolation=interp)

    def _render_silhouette(self, verts3d):
        """Soft silhouette of camera-frame vertices."""
        verts3d = verts3d @ self._flip_matrix.T
        mesh = Meshes(verts=[verts3d], faces=[self.obj_faces], textures=self._white_tex)
        silhouette_rgba = self.renderer(mesh, cameras=self.cameras)
        silhouette = silhouette_rgba[..., 3][0]
        silhouette = 1 - (1 - silhouette.clamp(0, 1)) ** 10   # push partial coverage toward 1
        return silhouette

    def _scale_loss(self, verts3d):
        """Box-size barrier against the target box."""
        _, bbox_mesh_size = get_2d_bbox_from_verts(verts3d, self.crop_fl,
                                                   self.crop["R"], self.crop["R"], pp=self.crop_pp)
        diff = bbox_mesh_size - self.bbox_img_size
        tol = 0.05 * torch.norm(self.bbox_img_size)
        return F.relu(torch.sum(diff ** 2) - tol ** 2)

    def _position_loss(self, verts3d):
        """Box-center barrier against the target box."""
        mesh_center, _ = get_2d_bbox_from_verts(verts3d, self.crop_fl,
                                                self.crop["R"], self.crop["R"], pp=self.crop_pp)
        target = torch.sum((mesh_center - self.bbox_center) ** 2)
        tol = 0.03 * torch.norm(self.bbox_img_size)
        return F.relu(target - tol ** 2)

    def _sil_loss(self, silhouette):
        """Symmetric silhouette loss."""
        gt_mask = self.gt_mask_norm
        gt_area = gt_mask.sum().clamp(min=1e-6)
        pred_area = silhouette.sum().clamp(min=1e-6)
        gt_not_covered = ((gt_mask * (1.0 - silhouette)).sum()) / gt_area
        pred_not_matched = ((silhouette * (1.0 - gt_mask)).sum()) / pred_area
        return ((gt_not_covered + pred_not_matched) * 100) ** 2

    def all_losses(self):
        """Return the (scale, position, silhouette, contour) losses."""
        verts3d = self.forward()
        scale_loss = self._scale_loss(verts3d)
        pos_loss = self._position_loss(verts3d)

        silhouette = self._render_silhouette(verts3d)
        sil_loss = self._sil_loss(silhouette)
        cont_loss = _contour_loss_dt(silhouette, self._gt_dt2, self._gt_edge)
        return scale_loss, pos_loss, sil_loss, cont_loss


@torch.no_grad()
def _precompute_contour_dt2(gt_mask, thr=0.01):
    """Squared distance to the target contour, and the contour mask."""
    gt_c = _soft_contour(gt_mask)
    edge = (gt_c > thr).detach().cpu().numpy().astype(np.uint8)
    if edge.sum() == 0:
        return torch.zeros_like(gt_mask), torch.zeros_like(gt_mask, dtype=torch.bool)
    dt = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 3).astype(np.float32)
    return (torch.tensor(dt, device=gt_mask.device) ** 2,
            torch.tensor(edge.astype(bool), device=gt_mask.device))


def _contour_loss_dt(pred_soft, dt2, gt_edge, cover_w=400.0):
    """Contour loss: distance to the target contour plus its coverage."""
    pred_c = _soft_contour(pred_soft)
    to_gt = (pred_c * dt2).sum() / (pred_c.sum() + 1e-8)
    pc_n = pred_c / (pred_c.max().detach() + 1e-8)
    cover = ((1.0 - pc_n[gt_edge]) ** 2).mean() if bool(gt_edge.any()) else pred_c.sum() * 0.0
    return to_gt + cover_w * cover


def _soft_contour(x, sigma=1.0):
    """Soft inner boundary relu(x - blur(x)) of an (H, W) map."""
    y = x.unsqueeze(0).unsqueeze(0)
    y = KF.GaussianBlur2d((5, 5), (sigma, sigma))(y)
    return F.relu(x - y.squeeze(0).squeeze(0))


def trans_to_rot(t: torch.Tensor) -> torch.Tensor:
    """Rotation taking the optical axis (+z) onto the direction of t."""
    x, y, z = t[..., 0], t[..., 1], t[..., 2]
    yaw = torch.atan2(x, z)
    pitch = -torch.atan2(y, torch.hypot(x, z))
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cx, sx = torch.cos(pitch), torch.sin(pitch)
    z0 = torch.zeros_like(cy)
    o1 = torch.ones_like(cy)
    Ry = torch.stack((
        torch.stack((cy, z0, sy), dim=-1),
        torch.stack((z0, o1, z0), dim=-1),
        torch.stack((-sy, z0, cy), dim=-1)
    ), dim=-2)
    Rx = torch.stack((
        torch.stack((o1, z0, z0), dim=-1),
        torch.stack((z0, cx, -sx), dim=-1),
        torch.stack((z0, sx, cx), dim=-1)
    ), dim=-2)
    return Ry @ Rx


def iou_of(model, verts3d):
    """Silhouette IoU of verts3d against the model's target mask."""
    with torch.no_grad():
        sil = model._render_silhouette(verts3d)
        gt = model.gt_mask_norm
        s = (sil > 0.5).float()
        inter = (s * gt).sum()
        union = ((s + gt) > 0.5).float().sum().clamp_min(1)
        return float(inter / union)


def _render_zbuf(model, verts3d):
    """Rendered front-surface depth of verts3d (-1 where empty)."""
    v = verts3d @ model._flip_matrix.T
    mesh = Meshes(verts=[v], faces=[model.obj_faces], textures=model._white_tex)
    return model.renderer.rasterizer(mesh, cameras=model.cameras).zbuf[0, ..., 0]


def object_depth_N(model, obj_verts_fit, depth):
    """Median depth / rendered-depth ratio over the target mask; 1.0 if no valid pixel."""
    depth_np = np.nan_to_num(depth.astype(np.float32))
    depth_np = model.crop_full_map(depth_np, nearest=True)
    depth_cam = torch.tensor(depth_np, device=obj_verts_fit.device)
    with torch.no_grad():
        zbuf = _render_zbuf(model, obj_verts_fit)
        gt = model.gt_mask_norm > 0.5
        valid = (zbuf > 0) & gt & torch.isfinite(depth_cam) & (depth_cam > 1e-6)
        if not bool(valid.any()):
            return 1.0
        return float((depth_cam[valid] / zbuf[valid]).median())
