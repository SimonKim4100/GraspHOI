"""Amodal object completion: Stable Diffusion inpainting, then rembg."""
import math
import os

import cv2
import numpy as np
from PIL import Image

from .depth_mask import behind_mask, depth_occluder_mask

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WALLPAPER = os.path.join(_HERE, "assets", "gray_wallpaper.jpeg")
SD15_INPAINT_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"

BAND_ITERS = 3        # 5x5 dilation steps of the occluder band
RIM_PX = 2            # object-side rim SD may repaint
CROP_PAD_FRAC = 0.5   # crop padding per side (fraction)


def _tile_to(bg, h, w):
    bh, bw = bg.shape[:2]
    if bh < h or bw < w:
        bg = np.tile(bg, (math.ceil(h / bh), math.ceil(w / bw), 1))
    return bg[:h, :w]


def _square_crop_box(region, pad_frac, H, W):
    """Padded square box (y0, y1, x0, x1) around region."""
    ys, xs = np.where(region)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    side = max(y1 - y0, x1 - x0)
    side += 2 * int(round(side * pad_frac))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, min(cy - side // 2, H - 1)); x0 = max(0, min(cx - side // 2, W - 1))
    return y0, min(H, y0 + side), x0, min(W, x0 + side)


class AmodalCompleter:
    """Amodal completer with a lazily loaded SD pipeline."""

    def __init__(self, model_id=SD15_INPAINT_MODEL, wallpaper_path=DEFAULT_WALLPAPER,
                 device="cuda", sd_size=512, steps=50, guidance=7.5, strength=1.0):
        self.model_id = model_id
        self.device = device
        self.sd_size = sd_size
        self.steps = steps
        self.guidance = guidance
        self.strength = strength    # 1.0 = regenerate entirely
        self.wallpaper = np.array(Image.open(wallpaper_path).convert("RGB"))
        self._pipe = None
        self._rembg = None

    def _get_pipe(self):
        """SD pipeline on the GPU, loaded on first use."""
        if self._pipe is None:
            import torch
            from diffusers import StableDiffusionInpaintPipeline
            local = os.path.isdir(str(self.model_id))
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id, torch_dtype=torch.float16, safety_checker=None,
                requires_safety_checker=False, local_files_only=local)
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pipe.enable_attention_slicing()
            self._pipe = pipe.to(self.device)
        else:
            self._pipe.to(self.device)
        return self._pipe

    def free(self):
        """Park the SD pipeline on the CPU."""
        if self._pipe is not None:
            import gc
            import torch
            try:
                self._pipe.to("cpu")
            except Exception:
                del self._pipe; self._pipe = None
            gc.collect(); torch.cuda.empty_cache()

    def _strip_bg(self, rgb):
        """Foreground alpha (HxW uint8) from rembg."""
        if self._rembg is None:
            from rembg import new_session
            self._rembg = new_session("u2net")
        from rembg import remove
        rgba = np.array(remove(Image.fromarray(rgb), session=self._rembg))
        a = rgba[..., 3].astype(np.float32) / 255.0
        return (a * 255).astype(np.uint8)

    def complete(self, image, object_mask, depth, prompt, *, seed=None):
        """Complete the visible object; image is RGB, depth in metres."""
        import torch
        H, W = object_mask.shape[:2]
        objb = np.asarray(object_mask)
        objb = (objb > 127) if objb.dtype != bool else objb
        image = np.ascontiguousarray(image[..., :3]).astype(np.uint8)

        # 1. occluder region
        occ = depth_occluder_mask(objb, depth)
        finite = np.isfinite(depth) & (depth > 1e-6)
        k = np.ones((5, 5), np.uint8)
        grown = cv2.dilate(occ, k, iterations=BAND_ITERS) > 0
        near = cv2.dilate(objb.astype(np.uint8), k, iterations=BAND_ITERS) > 0
        behind = behind_mask(objb, depth)
        occ = ((occ > 0) | (grown & near & finite & ~behind)) & ~objb
        # object-side rim
        rim_k = np.ones((2 * RIM_PX + 1, 2 * RIM_PX + 1), np.uint8)
        occ = occ | ((cv2.dilate(occ.astype(np.uint8), rim_k) > 0) & objb)

        # 2. square crop on gray wallpaper
        region = objb | occ
        y0, y1, x0, x1 = _square_crop_box(region, CROP_PAD_FRAC, H, W)
        cimg = image[y0:y1, x0:x1]
        cobj = objb[y0:y1, x0:x1]
        cocc = occ[y0:y1, x0:x1]
        ch, cw = cimg.shape[:2]
        bg = _tile_to(self.wallpaper, ch, cw)
        sd_in = cimg.copy()
        sd_in[~cobj] = bg[~cobj]

        # 3. SD inpainting
        S = self.sd_size
        sd_img = Image.fromarray(sd_in).resize((S, S), Image.LANCZOS)
        sd_occ = Image.fromarray((cocc.astype(np.uint8) * 255)).resize((S, S), Image.NEAREST)
        gen = torch.Generator(device=self.device).manual_seed(int(seed)) if seed is not None else None
        with torch.no_grad():
            out = self._get_pipe()(
                prompt=prompt, image=sd_img, mask_image=sd_occ,
                num_inference_steps=self.steps, guidance_scale=self.guidance,
                strength=self.strength, generator=gen,
            ).images[0]
        out = np.array(out.resize((cw, ch), Image.LANCZOS))

        # 4. rembg alpha + visible mask
        a = self._strip_bg(out)
        a = np.maximum(a, cobj.astype(np.uint8) * 255)
        af = a.astype(np.float32) / 255.0
        cutout = (out * af[..., None] + 255.0 * (1.0 - af[..., None])).astype(np.uint8)
        on_white = np.full((H, W, 3), 255, np.uint8); on_white[y0:y1, x0:x1] = cutout
        alpha = np.zeros((H, W), np.uint8); alpha[y0:y1, x0:x1] = a
        return dict(on_white=on_white, alpha=alpha)


def inpaint_prompt(object_name):
    """Inpainting prompt 'a <object>'."""
    return f"a {object_name.replace('_', ' ').strip() or 'object'}"
