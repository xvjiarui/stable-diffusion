from omegaconf import OmegaConf
from .util import instantiate_from_config
import os
from .models.diffusion.ddpm import LatentDiffusion

default_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/default.yaml")
def build_default_ldm() -> LatentDiffusion:

    config = OmegaConf.load(default_cfg_path)
    return instantiate_from_config(config.model)

def build_ldm_from_cfg(cfg_name="default.yaml") -> LatentDiffusion:

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", cfg_name)

    config = OmegaConf.load(cfg_path)
    return instantiate_from_config(config.model)
