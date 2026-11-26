"""GraspHOI: human, hand and object reconstruction from a single image."""
import argparse
import contextlib
import gc
import os
import sys
import traceback
import warnings
import logging
from collections import namedtuple

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch

# trusted checkpoints
_torch_load_orig = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_compat

from amodal import AmodalCompleter, inpaint_prompt
from find_contact.depth_seat import contact_region, detect_and_seat
from find_contact.hoi_detr_contact import assign_hand_boxes
from find_mano.hands_stage import estimate_hands, fitted_hands_only
from find_mano.mano import Wilor
from human_recon.constants import REPO_ROOT
from human_recon.estimator import HumanEstimator
from human_recon.person_camera import PersonCamera
from object_pose.depth_gauge import MIN_HAND_VERTS, body_depth_gauge, hand_depth_gauge
from object_pose.layout import as_numpy
from object_pose.placement import place_object
from object_pose.ppd_depth import predict_depth, park_cpu as _ppd_park
from object_recon.reconstruction import reconstruct_object
from optimizer.jointoptim import optimize_joint
from utils.export import export_sample
from utils.workers import HoiDetr, Sam3dBody, Sam3dObjects

AMODAL_SEED = 1234                    # Stable Diffusion seed
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
VISIBLE_MASK = "object_mask.png"      # visible-object mask file

Sample = namedtuple("Sample", "name image mask object")


@contextlib.contextmanager
def _sample_errors(sample, stage, failed, keep_going):
    """Log and record a failing sample if keep_going, else re-raise."""
    try:
        yield
    except Exception:
        if not keep_going:
            raise
        failed[sample.name] = stage
        print(f"[error] {sample.name}: {stage} failed\n{traceback.format_exc()}", flush=True)


class GraspHOIPipeline:
    """GraspHOI pipeline over one or more samples."""

    def __init__(self, hoidetr=HoiDetr, person_camera=PersonCamera, sam3d_body=Sam3dBody,
                 sam3d_objects=Sam3dObjects, amodal_completer=AmodalCompleter):
        self._make_hoidetr = hoidetr
        self._make_person_camera = person_camera
        self._make_sam3d_body = sam3d_body
        self._make_sam3d_objects = sam3d_objects
        self._make_amodal_completer = amodal_completer
        self._amodal_completer = None
        self.human = None
        self.wilor = None

    def _load_models(self):
        """SMPL/SMPL-H layers and WiLoR, loaded once."""
        if self.human is None:
            self.human = HumanEstimator()
            self.device = self.human.device
            self.smpl_model = self.human.smpl_model
            self.smplh_model = self.human.smplh_model
            self.wilor = Wilor()

    def _amodal(self):
        """Amodal completer, built on first use."""
        if self._amodal_completer is None:
            self._amodal_completer = self._make_amodal_completer()
        return self._amodal_completer

    def run(self, samples, out_dir, keep_going=False):
        """Reconstruct samples into out_dir/<name>/. Returns {name: failed stage}."""
        failed = {}
        inputs = self.compute_inputs(samples, failed, keep_going)
        pending = [s for s in samples if s.name in inputs]
        if not pending:
            return failed
        # start before the main models load
        with self._make_sam3d_objects() as objects:
            self._load_models()
            for k, s in enumerate(pending, 1):
                print(f"[sample] {s.name} ({k}/{len(pending)})", flush=True)
                with _sample_errors(s, "reconstruction", failed, keep_going):
                    self.process_sample(s, *inputs[s.name], objects, out_dir)
        return failed

    def compute_inputs(self, samples, failed, keep_going):
        """Run the input models. Returns {name: (record, person box, K, body fit)}."""
        records, cameras, bodies = {}, {}, {}
        with self._make_hoidetr() as hoidetr:
            for s in samples:
                with _sample_errors(s, "HOI-DETR", failed, keep_going):
                    record = hoidetr.detect(s.image)
                    n_hands = sum(d["class_id"] == 0 for d in record["detections"])
                    print(f"[inputs] {s.name}: HOI-DETR {len(record['detections'])} detection(s), "
                          f"{n_hands} hand(s), {len(record['hf'])} hand-object link(s)", flush=True)
                    records[s.name] = record
        pending = [s for s in samples if s.name in records]
        if not pending:
            return {}

        with self._make_person_camera() as person_camera:
            for s in pending:
                with _sample_errors(s, "person box and camera", failed, keep_going):
                    bbox, K = person_camera(_read_image(s.image))
                    print(f"[inputs] {s.name}: person box {np.round(bbox, 1).tolist()}, f = {K[0, 0]:.1f} px",
                          flush=True)
                    cameras[s.name] = (bbox, K)
        pending = [s for s in pending if s.name in cameras]
        if not pending:
            return {}

        with self._make_sam3d_body() as sam3d_body:
            for s in pending:
                with _sample_errors(s, "SAM 3D Body", failed, keep_going):
                    bodies[s.name] = sam3d_body.fit(s.image, *cameras[s.name])
                    print(f"[inputs] {s.name}: SAM 3D Body fit", flush=True)
        return {s.name: (records[s.name],) + cameras[s.name] + (bodies[s.name],)
                for s in pending if s.name in bodies}

    def process_sample(self, sample, record, bbox_xyxy, K, body, objects, out_dir):
        """Reconstruct one sample into out_dir/<name>/."""
        # FoundationPose sets a CUDA default
        torch.set_default_tensor_type(torch.FloatTensor)
        img_cv2 = _read_image(sample.image)
        vis_mask = _read_visible_mask(sample.mask, img_cv2.shape[:2])
        img_ext = os.path.splitext(sample.image)[1]

        self.human.set_sample(body, bbox_xyxy, K)
        bbox_center, bbox_scale = self.human.person_box()
        cam_int = self.human.cam_intrinsics()

        # PPD metric depth
        depth_map = predict_depth(img_cv2, cam_int)

        # initial body
        batch = self.human.body_batch(img_cv2, bbox_center, bbox_scale, cam_int)
        img_h, img_w = batch['img_size'][0]
        out_smpl_params, out_cam, focal_length_ = self.human.body_forward(batch)
        output_vertices, _, output_cam_trans = self.human.run_body_model(
            self.smpl_model, out_smpl_params, out_cam, batch)
        focal_length = (focal_length_[0], focal_length_[0])

        smpl_verts = (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
        smplh_faces = self.smplh_model.faces

        # align the depth map to the body
        s_body = body_depth_gauge(smpl_verts[0], depth_map, cam_int)
        if np.isfinite(s_body) and s_body > 1e-3:
            depth_map = depth_map * s_body
            print(f"[depth] depth map aligned to the body (scale {s_body:.3f})", flush=True)
        else:
            print(f"[depth] body alignment skipped (scale {s_body}); depth kept", flush=True)

        # hands: HOI-DETR -> WiLoR -> SMPL-H
        boxes, is_right = assign_hand_boxes(record, smpl_verts[0], focal_length, img_w, img_h)
        left_flag = 0 in is_right
        right_flag = 1 in is_right
        left_kpts = None
        right_kpts = None
        if left_flag or right_flag:
            smplh_params, smplh_verts, output_cam_trans, left_kpts, right_kpts = estimate_hands(
                self.wilor, self.human, img_cv2, boxes, is_right,
                left_flag, right_flag, out_smpl_params, out_cam, focal_length_, batch)
        else:
            smplh_params, smplh_verts, output_cam_trans = fitted_hands_only(
                self.human, out_smpl_params, out_cam, batch)

        # amodal completion
        prompt = inpaint_prompt(sample.object)
        amodal_out = self._amodal().complete(
            img_cv2[:, :, ::-1], vis_mask, depth_map, prompt, seed=AMODAL_SEED)
        amodal_on_white = amodal_out["on_white"]
        self._amodal().free()
        alpha = np.maximum(np.asarray(amodal_out["alpha"]), (vis_mask > 127).astype(np.uint8) * 255)
        amodal_sil = (alpha > 127).astype(np.uint8) * 255
        print(f"[amodal] '{prompt}': S_a {int((amodal_sil > 0).sum())} px, "
              f"S_vis {int((vis_mask > 127).sum())} px", flush=True)

        # free GPU memory
        self.wilor.model.to("cpu"); _ppd_park()
        gc.collect(); torch.cuda.empty_cache()

        # SAM 3D Objects mesh and layout
        obj_mesh_tm, layout = reconstruct_object(objects, img_cv2[:, :, ::-1], vis_mask, amodal_on_white,
                                                 amodal_sil)
        obj_faces = torch.as_tensor(np.asarray(obj_mesh_tm.faces), dtype=torch.long, device=self.device)

        # align the depth map to the hands
        gamma_h, n_hand_verts = hand_depth_gauge(as_numpy(depth_map, np.float32), smplh_verts, smplh_faces,
                                                 as_numpy(cam_int).reshape(3, 3), vis_mask,
                                                 left_flag, right_flag)
        depth_map = as_numpy(depth_map, np.float32) * gamma_h
        print(f"[depth] hand gauge gamma_h = {gamma_h:.4f} ({n_hand_verts} hand vertices"
              + ("" if n_hand_verts >= MIN_HAND_VERTS else f" < {MIN_HAND_VERTS}: body gauge kept") + ")",
              flush=True)

        # metric placement
        optimizer_model, verts_2d, _, _ = place_object(
            obj_mesh_tm, layout, obj_faces, cam_int, focal_length_, batch, img_cv2, amodal_on_white,
            amodal_sil, vis_mask, depth_map, smplh_verts, left_flag, right_flag, device=self.device)

        # contact and seating
        det, left_contact, right_contact, obj_verts, contact_pairs, _ = detect_and_seat(
            optimizer_model, verts_2d, depth_map, record, contact_region(amodal_on_white), vis_mask,
            smplh_verts, smplh_faces, obj_faces, focal_length, img_w, img_h, left_flag, right_flag)

        # joint refinement
        smplh_params, smplh_verts, obj_verts = optimize_joint(
            self.human, smplh_params, smplh_verts, obj_verts, obj_faces,
            left_contact, right_contact, left_kpts, right_kpts,
            contact_pairs, out_cam, batch, output_cam_trans, focal_length_, amodal_sil)

        # posed joints in the camera frame
        with torch.no_grad():
            _, joints, _ = self.human.run_body_model(self.smplh_model, smplh_params, out_cam, batch)
        smplh_joints = (joints + output_cam_trans.unsqueeze(1))[0].detach().cpu().numpy().astype(np.float32)

        export_sample(out_dir, sample.name, img_ext, img_cv2, cam_int, smplh_params, output_cam_trans[0],
                      smplh_verts, smplh_joints, smplh_faces, obj_verts, obj_faces, obj_mesh_tm.visual,
                      amodal_on_white, det, left_flag, right_flag, focal_length, img_w, img_h)


def _read_image(path):
    """BGR image at path; raises if it cannot be read."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read the image {path}")
    return img


def _read_visible_mask(path, hw):
    """Visible-object mask (grayscale, > 127 inside); raises if invalid."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cannot read the visible-object mask {path}")
    if mask.shape != tuple(hw):
        raise ValueError(f"the visible-object mask {path} is {mask.shape[1]}x{mask.shape[0]} px, "
                         f"the image {hw[1]}x{hw[0]} px")
    if not (mask > 127).any():
        raise ValueError(f"the visible-object mask {path} is empty (no pixel above 127)")
    return mask


def collect_samples(root):
    """List the sample folders root/<object>__<id>/ as Sample tuples."""
    samples = []
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        images = [os.path.join(folder, name + e) for e in IMAGE_EXTS
                  if os.path.isfile(os.path.join(folder, name + e))]
        mask = os.path.join(folder, VISIBLE_MASK)
        if not images or not os.path.isfile(mask):
            print(f"[samples] skipping {folder}: it needs {name}.jpg (or .jpeg, .png) and {VISIBLE_MASK}",
                  flush=True)
            continue
        samples.append(Sample(name, images[0], mask, name.split("__")[0]))
    if not samples:
        raise FileNotFoundError(f"no sample folder <object>__<id>/ holding <object>__<id>.jpg (or .jpeg, "
                                f".png) and {VISIBLE_MASK} in {root}")
    return samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="GraspHOI: human, hand and object reconstruction from an image, a mask of the visible "
                    "object and the object's name.")
    one = parser.add_argument_group("one sample")
    one.add_argument("--image", help="RGB image (.jpg, .jpeg or .png)")
    one.add_argument("--mask", help="mask of the visible object, white inside, same size as the image")
    one.add_argument("--object", help="object name, used only in the inpainting prompt 'a <object>'")
    parser.add_argument("--samples", metavar="ROOT",
                        help="reconstruct every sample folder ROOT/<object>__<id>/ that holds "
                             f"<object>__<id>.jpg (or .jpeg, .png) and {VISIBLE_MASK}; the object name is "
                             "<object>, underscores read as spaces")
    parser.add_argument("--out", required=True, help="results are written to OUT/<name>/")
    args = parser.parse_args()
    if (args.samples is None) == (args.image is None):
        parser.error("give either --image (with --mask and --object) or --samples")
    if args.image is not None:
        if args.mask is None or not (args.object or "").strip():
            parser.error("--image needs --mask and --object")
        for path in (args.image, args.mask):
            if not os.path.isfile(path):
                parser.error(f"{path} not found")
    return args


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    if args.samples is not None:
        samples, keep_going = collect_samples(os.path.abspath(args.samples)), True
    else:
        image = os.path.abspath(args.image)
        samples = [Sample(os.path.splitext(os.path.basename(image))[0], image, os.path.abspath(args.mask),
                          args.object.strip())]
        keep_going = False
    os.makedirs(out_dir, exist_ok=True)
    # WiLoR's data path is cwd-relative
    os.chdir(REPO_ROOT)

    failed = GraspHOIPipeline().run(samples, out_dir, keep_going=keep_going)
    if failed:
        print(f"[run] {len(failed)} of {len(samples)} sample(s) failed: "
              + ", ".join(f"{name} ({stage})" for name, stage in failed.items()), flush=True)
        sys.exit(1)
    print(f"[run] {len(samples)} sample(s) reconstructed into {out_dir}", flush=True)


if __name__ == '__main__':
    main()
