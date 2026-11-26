"""HOI-DETR worker: detections and hand-object links of an image.
    python -I workers/hoidetr_worker.py <reply_fd> <root> <config> <ckpt>"""
import importlib.util
import os
import sys

if len(sys.argv) != 5:
    sys.exit(__doc__)
REPLY_FD, HOIDETR_ROOT, CONFIG, CHECKPOINT = sys.argv[1:]
# HOI-DETR clone
sys.path[:0] = [os.path.join(HOIDETR_ROOT, "demo"), HOIDETR_ROOT]

import mmcv
from mmdet.apis import init_detector
from mmdet.datasets.pipelines import Compose

# registers the Co-DETR modules
from projects import *  # noqa: F401,F403

from configs import CLASS_NAMES
from helpers import call_interaction, find_interaction_branch, run_inference
from predictions_io import detections_record

SCORE_THR = 0.3     # min. detection score
NMS_IOU = 0.5       # soft-NMS IoU threshold
DEVICE = "cuda:0"


def _load_serve():
    """workers/serve.py, loaded by path."""
    spec = importlib.util.spec_from_file_location(
        "grasphoi_serve", os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.serve


# adapted from HOI-DETR demo/demo.py (https://github.com/AhmadDarKhalil/HOI-DETR), MIT
def main():
    model = init_detector(CONFIG, CHECKPOINT, device=DEVICE)
    model.eval()
    test_pipeline = Compose(model.cfg.data.test.pipeline)
    interaction_branch = find_interaction_branch(model.query_head)

    def detect(request):
        img_path = request["image"]
        orig_img = mmcv.imread(img_path)
        dets, embeds = run_inference(model, test_pipeline, img_path, device=DEVICE,
                                     class_names=CLASS_NAMES, score_thr=SCORE_THR, nms_iou=NMS_IOU)
        dets = dets or []
        hands = [d for d in dets if d['class_id'] == 0]
        firsts = [d for d in dets if d['class_id'] == 1]
        seconds = [d for d in dets if d['class_id'] == 2]
        hf_inters, fs_inters = [], []
        for h in hands:
            for fst in firsts:
                interacts, prob = call_interaction(interaction_branch,
                                                   embeds[h['query_idx']], embeds[fst['query_idx']])
                if interacts:
                    hf_inters.append((h, fst, prob))
        for fst in firsts:
            for so in seconds:
                interacts, prob = call_interaction(interaction_branch,
                                                   embeds[fst['query_idx']], embeds[so['query_idx']])
                if interacts:
                    fs_inters.append((fst, so, prob))
        height, width = orig_img.shape[:2]
        det_j, hf_j, fs_j = detections_record(dets, hf_inters, fs_inters)
        return {"record": {"file_name": os.path.basename(img_path), "width": width, "height": height,
                           "detections": det_j, "hf": hf_j, "fs": fs_j}}

    _load_serve()(detect, REPLY_FD)


if __name__ == "__main__":
    main()
