# GraspHOI: Full-Body 3D Human-Object Reconstruction with Articulated Hands from a Single In-the-Wild Image
[arXiv](https://arxiv.org/abs/2608.28386) | [Project Page](https://simonkim4100.github.io/GraspHOI/)

This is the official implementation of our paper GraspHOI.

Authors: Semin Kim, Haechan Shin, Jongyoo Kim

GraspHOI is a category-agnostic method that reconstructs full-body 3D human-object interaction from a single RGB image, a mask of the visible object and the object's name. It produces a SMPL-H human and a colored object mesh generated from the image, with the fingers optimized to grasp the object, all in one metric camera frame.

## Contents
- [Installation](#installation)
  - [Step 1: Code and environment](#step-1-code-and-environment)
  - [Step 2: Model weights](#step-2-model-weights)
  - [Step 3: SMPL, SMPL+H and MANO body models](#step-3-smpl-smplh-and-mano-body-models)
  - [Step 4: SAM 3D weights](#step-4-sam-3d-weights)
- [Run demo](#run-demo)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Citation](#citation)

## Updates
- Nov 2026, code released.
- Nov 2026, arXiv updated.
- Aug 28, 2026, arXiv released.

## Installation
Tested on Ubuntu 22.04 with Python 3.11, CUDA 12.8 and PyTorch 2.8.0. GPUs are RTX 4090, 5090, PRO5000.

### Step 1: Code and environment
Needs git, conda and the CUDA 12.8 toolkit. The script clones the third-party code into `third_party/`, applies our patches and creates the conda environment `grasphoi`. Re-running it skips what is already in place.
```bash
CUDA_HOME=/path/to/cuda-12.8 bash tools/install.sh
```

### Step 2: Model weights
These files need no account (about 14 GB). Re-running the script skips the files that are already in place.
```bash
conda activate grasphoi
bash tools/fetch_checkpoints.sh
```

| Model | Size | Destination |
|---|---|---|
| WiLoR hand model, its configs and the MANO mean parameters | 2.4 GB | `data/wilor_data/` |
| Detectron2 ViTDet-H person detector | 2.6 GB | `data/camerahmr/model_final_f05665.pkl` |
| FoundationPose refiner and scorer | 250 MB | `object_pose/foundationpose_runtime/weights/` |
| Pixel-Perfect Depth and MoGe-2 | 3.2 GB | `third_party/pixel-perfect-depth/checkpoints/` |
| HOI-DETR | 5.5 GB | `third_party/HOI-DETR/checkpoints/epoch_5.pth` |

The FoundationPose weights and `ppd_moge.pth` come from Google Drive, which can refuse downloads when its quota is reached (it resets daily). Run the script again later, or fetch them in a browser: the script prints how for FoundationPose, and [`ppd_moge.pth`](https://drive.google.com/file/d/1tabmcsbRVDKDfmO4KU1vOjurzN-wp0HV/view) goes in `third_party/pixel-perfect-depth/checkpoints/`.

### Step 3: SMPL, SMPL+H and MANO body models
These models need free accounts and cannot be redistributed, so you download and convert them yourself. GraspHOI uses the `smplx` Python package, but no SMPL-X model files.

| File | What it is | Where to get it |
|---|---|---|
| `SMPL_NEUTRAL.pkl` | neutral SMPL body model | [CameraHMR website](https://camerahmr.is.tue.mpg.de), with the script below |
| `cam_model_cleaned.ckpt` | HumanFoV camera model | [CameraHMR website](https://camerahmr.is.tue.mpg.de), with the script below |
| `MANO_LEFT.pkl`, `MANO_RIGHT.pkl` | MANO hand models | [MANO website](https://mano.is.tue.mpg.de), "Models & Code" (`mano_v1_2.zip`) |
| `neutral/model.npz` | neutral SMPL+H body model | [MANO website](https://mano.is.tue.mpg.de), "Extended SMPL+H model" (`smplh.tar.xz`, which has no top-level folder) |

**Download.** Register on both websites and accept their licenses. The script asks for your CameraHMR login and puts its two files in `data/`. Download the two MANO website archives yourself and extract them anywhere you like. `MANO_DIR` is the folder that holds `MANO_LEFT.pkl`, and `SMPLH_DIR` is the folder that holds `neutral/model.npz`:
```bash
bash tools/fetch_camerahmr_assets.sh

unzip /path/to/mano_v1_2.zip -d /path/to
mkdir -p /path/to/smplh && tar -xf /path/to/smplh.tar.xz -C /path/to/smplh

MANO_DIR=/path/to/mano_v1_2/models
SMPLH_DIR=/path/to/smplh
```

**Convert.** The original SMPL and MANO files contain chumpy objects. This script converts them to plain NumPy, without needing chumpy, and merges the SMPL+H body with the MANO hand parameters. Run it in the same shell, so that `MANO_DIR` and `SMPLH_DIR` are still set:
```bash
conda activate grasphoi
python tools/prepare_body_models.py --smpl data/models/SMPL/SMPL_NEUTRAL.pkl --mano-dir "$MANO_DIR" --smplh-dir "$SMPLH_DIR"
```

### Step 4: SAM 3D weights
Both are gated Hugging Face repositories (about 16 GB in total). Request access at [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3) and [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects) and wait until both requests are approved; until then the downloads are refused. Then create a token of type Read at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), log in with it and download them:
```bash
conda activate grasphoi
hf auth login
hf download facebook/sam-3d-body-dinov3 --local-dir third_party/sam-3d-body/checkpoints/sam-3d-body-dinov3
hf download facebook/sam-3d-objects --repo-type model --max-workers 1 \
    --local-dir third_party/sam-3d-objects/checkpoints/hf-download &&
    mv third_party/sam-3d-objects/checkpoints/hf-download/checkpoints third_party/sam-3d-objects/checkpoints/hf &&
    rm -rf third_party/sam-3d-objects/checkpoints/hf-download
```

### Check the file layout
After these steps, the files should be in place like this:
```
data/
├── camerahmr/
│   ├── cam_model_cleaned.ckpt
│   └── model_final_f05665.pkl
├── models/SMPL/
│   ├── SMPL_NEUTRAL.pkl
│   └── SMPLH_NEUTRAL.pkl
└── wilor_data/
    ├── model_config.yaml
    ├── dataset_config.yaml
    ├── wilor_ckpts/checkpoints/wilor_final.ckpt
    └── mano/
        ├── MANO_RIGHT.pkl
        └── mano_mean_params.npz
object_pose/foundationpose_runtime/weights/
├── 2023-10-28-18-33-37/{config.yml, model_best.pth}   # refiner
└── 2024-01-11-20-02-45/{config.yml, model_best.pth}   # scorer
third_party/
├── HOI-DETR/checkpoints/epoch_5.pth
├── pixel-perfect-depth/checkpoints/{ppd_moge.pth, moge2.pt}
├── sam-3d-body/checkpoints/sam-3d-body-dinov3/{model.ckpt, assets/mhr_model.pt, ...}
└── sam-3d-objects/checkpoints/hf/{pipeline.yaml, ss_generator.ckpt, slat_generator.ckpt, ...}
```

The first run also downloads about 7 GB of models into the Hugging Face, torch hub and rembg caches, so it needs network access.

## Run demo

### Prepare data
Each sample needs:
- **An RGB image** (`.jpg`, `.jpeg` or `.png`) of a person with the object.
- **A mask of the visible object**: white inside, same size as the image. It can come from any segmenter. Keep hands and anything else in front of the object out of the mask; GraspHOI completes the hidden part itself.
- **The object's name**, used only as the text prompt for completing the hidden part (`"a <object>"`).

If you need test images, you can get some from [PICO-db](https://pico.is.tue.mpg.de).

### One image
```bash
conda activate grasphoi
python demo.py --image photo.jpg --mask photo_mask.png --object "coffee mug" --out outputs
```

### Many images
Put each sample in its own folder. The part of the folder name before `__` is the object name, with underscores read as spaces, so `coffee_mug__01` gives the prompt "a coffee mug":
```
inputs/
└── coffee_mug__01/
    ├── coffee_mug__01.jpg     # RGB image, named like the folder
    └── object_mask.png        # mask of the visible object
```
Then run:
```bash
python demo.py --samples inputs --out outputs
```
A reconstruction takes about two minutes per image on an RTX 4090. Loading the models adds a few minutes to each call, so process many images with one `--samples` call.

### Results
Each sample is written to `outputs/<name>/`, `<name>` being the image file name without extension:
```
outputs/<name>/
├── <name>_mesh.glb        # human and object as one scene
├── <name>_smplh.glb       # human mesh
├── <name>_object.glb      # object mesh with vertex colors
├── <name>_overlay.jpg     # reconstruction rendered over the input image
├── <name>_whitebg.jpg     # the same render on a white background
├── <name>_sides.jpg       # left, right and top views
├── <name>_amodal.png      # the object with its hidden parts completed
├── <name>_contact.json    # per-hand contact record
└── <name>_result.npz      # reconstruction in the camera frame
```
The renders take the extension of the input image.

## Acknowledgements
This project is built on top of these amazing research projects:
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) and [MHR](https://github.com/facebookresearch/MHR) for the human body.
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) for the object mesh and its initial pose and scale.
- [CameraHMR](https://github.com/pixelite1201/CameraHMR) for the camera intrinsics.
- [HOI-DETR](https://github.com/AhmadDarKhalil/HOI-DETR) for hand detection and hand-object interaction links.
- [WiLoR](https://github.com/rolpotamias/WiLoR) for hand reconstruction.
- [Pixel-Perfect Depth](https://github.com/gangweix/pixel-perfect-depth) and [MoGe](https://github.com/microsoft/MoGe) for metric depth.
- [Stable Diffusion 1.5 inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) for completing the hidden part of the object.
- [FoundationPose](https://github.com/NVlabs/FoundationPose) for object pose registration.
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast) for rasterization and [smplx](https://github.com/vchoutas/smplx) for the body model layers.

## License
GraspHOI is released under the [Apache License 2.0](LICENSE). Code and assets from other projects keep their own licenses: see the LICENSE files in `find_mano/wilor`, `object_pose/foundationpose_runtime`, `third_party/meta_mhr_conversion` and `amodal/assets`, and the NOTICE files in `find_mano/wilor` and `amodal/assets`. The patches in `patches/` change the third-party repositories cloned in Step 1, which keep their own licenses.

## Citation
```bibtex
@article{kim2026grasphoi,
    title = {{GraspHOI}: Full-Body {3D} Human-Object Reconstruction with Articulated Hands from a Single In-the-Wild Image},
    author = {Kim, Semin and Shin, Haechan and Kim, Jongyoo},
    journal = {arXiv preprint arXiv:2608.28386},
    year = {2026},
}
```
