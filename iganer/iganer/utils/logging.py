"""W&B / console logger for IGANER.

Adds, vs the original:
  - define_metric() calls so W&B knows to MAXIMIZE val_auc and MINIMIZE val_eer
    (shows the right "best" in the run summary)
  - log_epoch() helper that frames progress as "Epoch X/Y" and logs all the
    standard validation metrics together, keyed to the epoch step
  - graceful console fallback when W&B is disabled or unavailable
"""


class Logger:
    def __init__(self, cfg):
        self.enabled = bool(cfg.wandb.enabled)
        self.run = None
        self.wandb = None
        if self.enabled:
            try:
                import wandb
                from omegaconf import OmegaConf
                self.wandb = wandb
                self.run = wandb.init(
                    project=cfg.wandb.project,
                    entity=cfg.wandb.get("entity", None),
                    mode=cfg.wandb.get("mode", "online"),
                    name=cfg.wandb.get("name", None),
                    group=cfg.wandb.get("group", None),
                    tags=list(cfg.wandb.get("tags", []) or []),
                    notes=cfg.wandb.get("notes", None),
                    config=OmegaConf.to_container(cfg, resolve=True),
                )
                self._define_metrics()
            except Exception as e:
                print(f"[Logger] W&B off ({e}).", flush=True)
                self.enabled = False

    def _define_metrics(self):
        """Tell W&B how to summarize each metric (best = max or min)."""
        if not (self.enabled and self.run):
            return
        try:
            w = self.wandb
            # use epoch as the x-axis for epoch-level metrics
            w.define_metric("epoch")
            w.define_metric("val_auc",      step_metric="epoch", summary="max")
            w.define_metric("val_eer",      step_metric="epoch", summary="min")
            w.define_metric("val_loss",     step_metric="epoch", summary="min")
            w.define_metric("val_accuracy", step_metric="epoch", summary="max")
            w.define_metric("train_loss",   step_metric="epoch", summary="min")
            w.define_metric("best_metric",  step_metric="epoch")
            w.define_metric("learning_rate", step_metric="epoch")
        except Exception as e:
            print(f"[Logger] define_metric skipped ({e}).", flush=True)

    def log(self, d, step=None):
        """Low-level log: to W&B if enabled, else formatted console print."""
        if self.enabled and self.run:
            self.wandb.log(d, step=step)
        else:
            m = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in d.items())
            print(f"[{step}] {m}", flush=True)

    def log_epoch(self, epoch, total_epochs, metrics, tag="run",
                  best_metric=None, best_name=None, lr=None,
                  ckpt_path=None, train_loss=None):
        """Log a full epoch summary with clear progress framing.

        metrics: dict that may contain val_auc, val_eer, val_loss, accuracy ...
        """
        payload = {
            "tag": tag,
            "epoch": int(epoch),
            "total_epochs": int(total_epochs),
            "epoch_progress": float(epoch) / max(total_epochs, 1),
        }
        for k, v in (metrics or {}).items():
            if v is not None:
                payload[k] = v
        if train_loss is not None:
            payload["train_loss"] = float(train_loss)
        if lr is not None:
            payload["learning_rate"] = float(lr)
        if best_metric is not None:
            payload["best_metric"] = float(best_metric)
        if ckpt_path:
            payload["ckpt_path"] = str(ckpt_path)

        # always print a clear human-readable epoch line
        bm = (f" best({best_name})={best_metric:.4f}"
              if best_metric is not None and best_name else "")
        vl = metrics.get("val_loss") if metrics else None
        au = metrics.get("val_auc") if metrics else None
        ee = metrics.get("val_eer") if metrics else None
        line = (f"[{tag}] Epoch {epoch}/{total_epochs} | "
                f"train_loss={train_loss if train_loss is not None else float('nan'):.4f} | "
                f"val_loss={vl if vl is not None else float('nan'):.4f} | "
                f"AUC={au if au is not None else float('nan'):.4f} | "
                f"EER={ee if ee is not None else float('nan'):.4f}"
                f"{bm}")
        print(line, flush=True)

        # W&B: log keyed to epoch (not global step) so curves are clean
        if self.enabled and self.run:
            self.wandb.log(payload)

    def log_table(self, name, cols, data):
        if self.enabled and self.run:
            self.wandb.log({name: self.wandb.Table(columns=cols, data=data)})

    def finish(self):
        if self.enabled and self.run:
            self.wandb.finish()