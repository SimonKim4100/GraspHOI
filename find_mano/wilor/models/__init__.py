from pathlib import Path
from .wilor import WiLoR
from .mano_wrapper import MANO
from ..utils.download import cache_url

CACHE_DIR = "./data/wilor_data"
DEFAULT_CHECKPOINT = f'{CACHE_DIR}/wilor_ckpts/checkpoints/wilor_final.ckpt'
DEFAULT_CONFIG = f'{CACHE_DIR}/model_config.yaml'
DEFAULT_URL = "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt"

def download_checkpoint(
    url: str = DEFAULT_URL,
    out_path: Path | str = DEFAULT_CHECKPOINT,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        cache_url(url, str(out_path))
    return out_path

def load_wilor(checkpoint_path, cfg_path):
    from find_mano.wilor.configs import get_config
    print('Loading ', checkpoint_path)
    model_cfg = get_config(cfg_path, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if ('vit' in model_cfg.MODEL.BACKBONE.TYPE) and ('BBOX_SHAPE' not in model_cfg.MODEL):
        
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()
        
        # Update config to be compatible with demo

    model = WiLoR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg
