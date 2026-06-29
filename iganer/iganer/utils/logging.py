class Logger:
    def __init__(self, cfg):
        self.enabled=bool(cfg.wandb.enabled); self.run=None
        if self.enabled:
            try:
                import wandb; from omegaconf import OmegaConf
                self.wandb=wandb
                self.run=wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                    mode=cfg.wandb.mode, name=cfg.wandb.get("name",None),
                    config=OmegaConf.to_container(cfg, resolve=True))
            except Exception as e:
                print(f"[Logger] W&B off ({e})."); self.enabled=False
    def log(self, d, step=None):
        if self.enabled and self.run: self.wandb.log(d, step=step)
        else:
            m=" ".join(f"{k}={v:.4f}" if isinstance(v,float) else f"{k}={v}" for k,v in d.items())
            print(f"[{step}] {m}")
    def log_table(self, name, cols, data):
        if self.enabled and self.run: self.wandb.log({name:self.wandb.Table(columns=cols,data=data)})
    def finish(self):
        if self.enabled and self.run: self.wandb.finish()
