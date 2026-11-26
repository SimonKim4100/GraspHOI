"""SAM 3D Objects worker: mesh and layout of an object.
    python -I workers/sam3d_objects_worker.py <reply_fd> <root>"""
import functools
import importlib.util
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# read by SAM 3D's notebook/inference.py on import
os.environ.setdefault("CONDA_PREFIX", sys.prefix)

import numpy as np
import torch

MIN_MASK_PX = 16    # smallest object mask (pixels)


def _load_serve():
    """workers/serve.py, loaded by path."""
    spec = importlib.util.spec_from_file_location(
        "grasphoi_serve", os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.serve


def stagewise_offload(pipe):
    """Keep idle modules on the CPU; each stage moves its own to the GPU."""
    E, M = pipe.condition_embedders, pipe.models
    groups = {'compute_pointmap': [pipe.depth_model.model],
              'sample_sparse_structure': [M['ss_generator'], M['ss_decoder'], E['ss_condition_embedder']],
              'sample_slat': [M['slat_generator'], E['slat_condition_embedder']],
              'decode_slat': [M['slat_decoder_mesh'], M['slat_decoder_gs']]}
    for mods in list(groups.values()) + [[M['slat_decoder_gs_4'], M['ss_encoder']]]:
        for m in mods:
            if m is not None:
                m.to('cpu')
    torch.cuda.empty_cache()
    for name, mods in groups.items():
        orig = getattr(pipe, name)

        def wrapped(*a, _orig=orig, _mods=mods, **k):
            for m in _mods:
                m.to('cuda')
            try:
                return _orig(*a, **k)
            finally:
                for m in _mods:
                    m.to('cpu')
                torch.cuda.empty_cache()
        setattr(pipe, name, functools.wraps(orig)(wrapped))


def load_pipeline(repo):
    """SAM 3D Objects' Inference from <repo>, with the stagewise offload."""
    config = os.path.join(repo, 'checkpoints', 'hf', 'pipeline.yaml')
    if not os.path.isfile(config):
        raise FileNotFoundError(f"SAM 3D Objects checkpoints not found: {config} (see README, step 4)")
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        spec = importlib.util.spec_from_file_location('sam3d_nb_inference',
                                                      os.path.join(repo, 'notebook', 'inference.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        inf = mod.Inference(config, compile=False)
    finally:
        os.chdir(cwd)
    stagewise_offload(inf._pipeline)
    return inf


def run_one(inf, image, mask, seed):
    """Run the pipeline on a uint8 RGB image and its mask."""
    m = np.asarray(mask).astype(bool)
    if not (image.dtype == np.uint8 and image.ndim == 3 and m.shape == image.shape[:2]):
        raise ValueError(f"expected an (H, W, 3) uint8 image and an (H, W) mask, got {image.shape} "
                         f"{image.dtype} and {m.shape}")
    if m.sum() < MIN_MASK_PX:
        raise ValueError(f"object mask has {int(m.sum())} pixels (< {MIN_MASK_PX})")
    with torch.no_grad():
        out = inf(image[..., :3], m, seed=seed)
    mesh = out['mesh'][0]
    H, W = image.shape[:2]
    # normalized intrinsics -> pixel K
    Kn = out['intrinsics'].detach().double().cpu().numpy()
    K = np.array([[Kn[0, 0] * W, 0.0, Kn[0, 2] * W - 0.5],
                  [0.0, Kn[1, 1] * H, Kn[1, 2] * H - 0.5], [0.0, 0.0, 1.0]])
    rgb = (mesh.vertex_attrs[:, :3].detach().float().clamp(0, 1).cpu().numpy()
           if mesh.vertex_attrs is not None else np.full((len(mesh.vertices), 3), 0.5, np.float32))
    res = dict(V_canon=mesh.vertices.detach().float().cpu().numpy(),
               F=mesh.faces.detach().cpu().numpy().astype(np.int32),
               rgb=(rgb * 255).round().astype(np.uint8),
               q_wxyz=out['rotation'].detach().double().cpu().numpy().reshape(4),
               t_p3d=out['translation'].detach().double().cpu().numpy().reshape(3),
               s=float(out['scale'].reshape(-1)[0]), K=K, hw=np.array([H, W]))
    del out
    torch.cuda.empty_cache()
    return res


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    reply_fd, repo = sys.argv[1], os.path.abspath(sys.argv[2])
    inf = load_pipeline(repo)

    def reconstruct(request):
        with np.load(request["input"], allow_pickle=False) as z:
            res = run_one(inf, z["image"], z["mask"], seed=int(z["seed"]))
        out = request["output"]
        np.savez_compressed(out + '.tmp.npz', **res)
        os.replace(out + '.tmp.npz', out)
        return {}

    _load_serve()(reconstruct, reply_fd)


if __name__ == "__main__":
    main()
