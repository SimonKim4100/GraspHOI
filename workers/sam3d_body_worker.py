"""SAM 3D Body worker: MHR mesh -> SMPL-H fit.
    python -I workers/sam3d_body_worker.py <reply_fd> <root> <ckpt> <mhr> <smplh> <fitter> <mapping>"""
import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

# trusted checkpoints
_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{**kwargs, "weights_only": False})

FITTER_SHA256 = "2eb4fab49a16205b5eefe781b1d9300fc0afb8ec4b3a4ec02a4884097b6ca26c"
MAPPING_SHA256 = "4bfe2fd6abb598f38ba3a3aa5506ac45b4efb3cdaed7fe4757cfbbf20bbe6ac7"
FIT_PARAMS = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "betas")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_vendored_files(fitter_path, mapping_path):
    """Check the vendored Meta files against their SHA-256 digests."""
    expected = {fitter_path: FITTER_SHA256, mapping_path: MAPPING_SHA256}
    for path, expected_digest in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required official MHR conversion file is missing: {path}")
        actual_digest = _sha256(path)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"Official MHR conversion file failed SHA-256 verification: {path}\n"
                f"expected {expected_digest}\nactual   {actual_digest}")


def _unavailable(*_args, **_kwargs):
    raise RuntimeError("Unused by the MHR-vertex to SMPL-H fitting path")


def load_official_fitter(fitter_path):
    """Load Meta's PyTorchSMPLFitting with its unused imports stubbed."""
    shims = {}
    mhr_package = types.ModuleType("mhr")
    mhr_package.__path__ = []
    mhr_module = types.ModuleType("mhr.mhr")
    mhr_module.MHR = type("MHR", (), {})
    mhr_package.mhr = mhr_module
    shims["mhr"] = mhr_package
    shims["mhr.mhr"] = mhr_module

    pymomentum_module = types.ModuleType("pymomentum_fitting")
    pymomentum_module._NUM_BODY_BLENDSHAPES = 0
    pymomentum_module._NUM_HAND_BLENDSHAPES = 0
    pymomentum_module._NUM_HEAD_BLENDSHAPES = 0
    shims["pymomentum_fitting"] = pymomentum_module

    utils_module = types.ModuleType("utils")
    utils_module._concat_mhr_lbs_model_parameters = _unavailable
    utils_module.ChunkedSequence = type("ChunkedSequence", (), {})
    utils_module.compute_edge_loss = _unavailable
    utils_module.compute_edge_vectors = _unavailable
    utils_module.compute_vertex_loss = _unavailable
    utils_module.evaluate_model_fitting_error = _unavailable
    utils_module.get_batched_parameters = _unavailable
    utils_module.load_head_vertex_weights = _unavailable
    shims["utils"] = utils_module

    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in shims}
    try:
        sys.modules.update(shims)
        spec = importlib.util.spec_from_file_location("grasphoi_meta_mhr_pytorch_fitting", fitter_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load official fitter: {fitter_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return module.PyTorchSMPLFitting


def transfer_mhr_to_smplh_topology(vertices, mhr_faces, triangle_ids, barycentric_coordinates):
    """Barycentric transfer of MHR vertices onto the SMPL-H mesh."""
    triangle_vertices = vertices[:, mhr_faces[triangle_ids]]
    return (triangle_vertices * barycentric_coordinates[None, :, :, None]).sum(dim=2)


def mesh_edges(faces, device):
    """Unique undirected edges (E, 2) of a triangle mesh."""
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    return torch.as_tensor(np.unique(edges, axis=0), dtype=torch.long, device=device)


def _load_serve():
    """workers/serve.py, loaded by path."""
    spec = importlib.util.spec_from_file_location(
        "grasphoi_serve", os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.serve


def main():
    if len(sys.argv) != 8:
        sys.exit(__doc__)
    reply_fd, sam3d_root, checkpoint, mhr_model, smplh_model = sys.argv[1:6]
    fitter_path, mapping_path = Path(sys.argv[6]), Path(sys.argv[7])

    verify_vendored_files(fitter_path, mapping_path)
    device = torch.device("cuda")

    sys.path.insert(0, os.path.abspath(sam3d_root))
    import smplx
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    model, cfg = load_sam_3d_body(checkpoint, device=str(device), mhr_path=mhr_model)
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    # MHR faces and Meta's surface mapping
    mhr_faces = model.head_pose.faces.detach().to(device=device, dtype=torch.long)
    mapping = np.load(mapping_path)
    triangle_ids = torch.as_tensor(mapping["triangle_ids"], dtype=torch.long, device=device)
    barycentric_coordinates = torch.as_tensor(mapping["baryc_coords"], dtype=torch.float32, device=device)
    if int(triangle_ids.max()) >= len(mhr_faces):
        raise RuntimeError("Official MHR mapping is incompatible with SAM 3D Body's MHR faces")

    layer = smplx.SMPLH(
        model_path=smplh_model,
        gender="neutral",
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        batch_size=1,
    ).to(device)
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    edges = mesh_edges(np.asarray(layer.faces, dtype=np.int64), device)
    fitter_class = load_official_fitter(fitter_path)

    def fit(request):
        intrinsics = torch.as_tensor(np.asarray(request["cam_int"], dtype=np.float32),
                                     dtype=torch.float32)[None]
        bbox = np.asarray(request["bbox_xyxy"], dtype=np.float32)[None]
        result = estimator.process_one_image(
            request["image"],
            bboxes=bbox,
            cam_int=intrinsics,
            inference_type="body",
            use_mask=False,
        )
        if not result:
            raise RuntimeError("no SAM 3D Body output")
        person = result[0]
        # camera-space MHR vertices
        camera_vertices = [person["pred_vertices"] + person["pred_cam_t"][None]]
        mhr_vertices = torch.as_tensor(np.stack(camera_vertices), dtype=torch.float32, device=device)
        target = transfer_mhr_to_smplh_topology(
            mhr_vertices, mhr_faces, triangle_ids, barycentric_coordinates)
        fitter = fitter_class(
            smpl_model=layer,
            smpl_edges=edges,
            # same body + hand interface as SMPL-H
            smpl_model_type="smplx",
            hand_pose_dim=45,
            device=str(device),
            batch_size=1,
        )
        # fit() recenters its target in place
        fitted = fitter.fit(target_vertices=target.clone(), single_identity=False, is_tracking=False)
        out = {k: fitted[k][0].detach().cpu().numpy().astype(np.float32).tolist() for k in FIT_PARAMS}
        out["trans"] = fitted["transl"][0].detach().cpu().numpy().astype(np.float32).tolist()
        return out

    _load_serve()(fit, reply_fd)


if __name__ == "__main__":
    main()
