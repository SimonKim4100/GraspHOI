# Adapted from CLIFF, common/renderer_pyrd.py (https://github.com/huawei-noah/noah-research/tree/master/CLIFF),
# MIT License:
#
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Pyrender body and object views."""
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'   # must be set before pyrender loads
import trimesh
import pyrender
import numpy as np
import colorsys
import cv2


class Renderer(object):
    """Offscreen renderer with a pinhole camera at the origin."""

    def __init__(self, focal_length=600, img_w=512, img_h=512):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                                   viewport_height=img_h,
                                                   point_size=1.0)
        self.camera_center = [img_w // 2, img_h // 2]
        self.focal_length = focal_length

    def render_front_view_textured(self, human_verts, human_faces, obj_trimesh, bg_img_bgr):
        """Render the body and textured object over the BGR background."""
        scene = pyrender.Scene(bg_color=(1., 1., 1., 1.), ambient_light=np.ones(3) * 0.45)
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                  cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=np.eye(4))
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0]))
        scene.add(light, pose=trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0]))
        # OpenCV -> OpenGL camera frame
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])

        hmesh = trimesh.Trimesh(human_verts, human_faces, process=False)
        hmesh.apply_transform(rot)
        human_color = colorsys.hsv_to_rgb(0.6, 0.4, 0.8)
        hmat = pyrender.MetallicRoughnessMaterial(metallicFactor=0.2, alphaMode='OPAQUE',
                                                  baseColorFactor=human_color)
        scene.add(pyrender.Mesh.from_trimesh(hmesh, material=hmat, wireframe=False), 'human')

        omesh = obj_trimesh.copy()
        omesh.apply_transform(rot)
        scene.add(pyrender.Mesh.from_trimesh(omesh), 'object')

        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_bgr = color_rgba[:, :, :3][:, :, ::-1]
        # covered pixels: 0.9 render + 0.1 background; the rest keeps the background
        covered = (depth_map > 0)[:, :, None].astype(np.float32)
        render_w = 0.9
        img = color_bgr * covered * render_w + bg_img_bgr * (1 - covered)
        return img + covered * bg_img_bgr * (1 - render_w)

    @staticmethod
    def _look_at(eye, target, up):
        """Camera-to-world pose looking from `eye` at `target`."""
        eye = np.asarray(eye, np.float64); target = np.asarray(target, np.float64)
        f = target - eye; f /= (np.linalg.norm(f) + 1e-9)          # forward
        s = np.cross(f, up); s /= (np.linalg.norm(s) + 1e-9)        # right
        u = np.cross(s, f)                                          # up
        pose = np.eye(4)
        pose[:3, 0] = s; pose[:3, 1] = u; pose[:3, 2] = -f; pose[:3, 3] = eye
        return pose

    def render_side_triplet(self, human_mesh, obj_mesh):
        """Left, right and top views of the upright meshes, as one BGR image."""
        size, gap = 512, 12                                          # panel side, separator width (px)
        allv = np.vstack([np.asarray(human_mesh.vertices), np.asarray(obj_mesh.vertices)])
        c = allv.mean(0)
        r = float(np.linalg.norm(allv - c, axis=1).max()) + 1e-6
        f = float(size)
        d = f * r / (size / 2.0) * 1.15                             # bounding sphere + 15% margin
        human_color = colorsys.hsv_to_rgb(0.6, 0.4, 0.8)
        # (name, camera position, camera up)
        views = [
            ('left',  c + np.array([ d, 0., 0.]), np.array([0., 1., 0.])),
            ('right', c + np.array([-d, 0., 0.]), np.array([0., 1., 0.])),
            ('up',    c + np.array([0.,  d, 0.]), np.array([0., 0., 1.])),
        ]
        hmat = pyrender.MetallicRoughnessMaterial(metallicFactor=0.2, alphaMode='OPAQUE',
                                                  baseColorFactor=human_color)
        # body without vertex colors
        hflat = trimesh.Trimesh(np.asarray(human_mesh.vertices), np.asarray(human_mesh.faces),
                                process=False)
        rr = pyrender.OffscreenRenderer(viewport_width=size, viewport_height=size, point_size=1.0)
        panels = []
        for _, eye, up in views:
            scene = pyrender.Scene(bg_color=(1., 1., 1., 1.), ambient_light=np.ones(3) * 0.5)
            cam = pyrender.camera.IntrinsicsCamera(fx=f, fy=f, cx=size / 2.0, cy=size / 2.0,
                                                   znear=0.01, zfar=max(1000.0, 4 * d))
            pose = self._look_at(eye, c, up)
            scene.add(cam, pose=pose)
            scene.add(pyrender.DirectionalLight(color=[1., 1., 1.], intensity=3.0), pose=pose)
            scene.add(pyrender.Mesh.from_trimesh(hflat, material=hmat))
            scene.add(pyrender.Mesh.from_trimesh(obj_mesh))
            color_rgba, _ = rr.render(scene, flags=pyrender.RenderFlags.RGBA)
            panels.append(color_rgba[:, :, :3][:, :, ::-1])        # RGB -> BGR
        rr.delete()
        panels = [self._crop_pad_square(p, out=size) for p in panels]
        sep = np.full((size, gap, 3), 255, np.uint8)
        stacked = []
        for i, p in enumerate(panels):
            stacked.append(p.astype(np.uint8))
            if i < len(panels) - 1:
                stacked.append(sep)
        return np.concatenate(stacked, axis=1)

    @staticmethod
    def _crop_pad_square(img, out=512, pad_frac=0.08):
        """Crop a render to its content and white-pad it to a square."""
        nonwhite = np.any(img < 245, axis=2)
        if not nonwhite.any():
            return cv2.resize(img, (out, out))
        ys, xs = np.where(nonwhite)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        crop = img[y0:y1 + 1, x0:x1 + 1]
        h, w = crop.shape[:2]
        side = int(max(h, w) * (1 + 2 * pad_frac))
        canvas = np.full((side, side, 3), 255, np.uint8)
        oy, ox = (side - h) // 2, (side - w) // 2
        canvas[oy:oy + h, ox:ox + w] = crop
        return cv2.resize(canvas, (out, out))

    def delete(self):
        """Release the offscreen context."""
        self.renderer.delete()
