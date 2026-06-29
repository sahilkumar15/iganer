from omegaconf import OmegaConf
def load_config(default_path, exp_path=None, overrides=None):
    cfg=OmegaConf.load(default_path)
    if exp_path: cfg=OmegaConf.merge(cfg, OmegaConf.load(exp_path))
    if overrides: cfg=OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
