"""Generalized checkpoint management for IGANER.

A single CheckpointManager handles, per experiment/variant:
  - periodic checkpoints (every N epochs, configurable)
  - top-K best checkpoints by a configurable metric (max or min mode)
  - a rolling latest.pth written every epoch (for auto-resume)
  - automatic pruning of worse-than-top-K checkpoints
  - full training state: model, optimizer, scheduler, AMP scaler, epoch,
    global step, best-metric history, config, and RNG states

Design goals: modular, generalized (not hardcoded to one metric or model),
atomic writes (no corruption on interruption), and clear filenames.

Usage
-----
    ckpt_mgr = CheckpointManager(
        ckpt_dir="outputs/ckpt/v8_iganer",
        metric_name="val_auc", metric_mode="max",
        save_every=10, top_k=3)

    # resume (returns dict or None)
    state = ckpt_mgr.maybe_resume(resume="auto")
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]

    # each epoch
    ckpt_mgr.save(epoch, global_step, metric_value,
                  model=model, optimizer=optimizer,
                  scheduler=scheduler, scaler=scaler, config=cfg)
"""
import os
import glob
import shutil
import random
import tempfile
import numpy as np
import torch


class CheckpointManager:
    """Handles periodic + top-K + latest checkpointing with auto-resume."""

    def __init__(self, ckpt_dir, metric_name="val_auc", metric_mode="max",
                 save_every=10, top_k=3, verbose=True):
        """
        Args:
            ckpt_dir    : directory for this experiment/variant's checkpoints
            metric_name : key of the validation metric to rank by (e.g. val_auc)
            metric_mode : "max" (higher is better) or "min" (lower is better)
            save_every  : also save a periodic checkpoint every N epochs
            top_k       : how many best checkpoints to keep
            verbose     : print on save/prune/resume
        """
        assert metric_mode in ("max", "min"), f"bad metric_mode: {metric_mode}"
        self.dir = ckpt_dir
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.save_every = int(save_every)
        self.top_k = int(top_k)
        self.verbose = verbose
        os.makedirs(self.dir, exist_ok=True)

        # in-memory registry of best checkpoints: list of (metric, path)
        self.best_ckpts = []
        # full best-metric history across epochs: list of (epoch, metric)
        self.history = []
        # rebuild registry if resuming into an existing dir
        self._rescan_existing()

    # ──────────────────────────────────────────────────────────────────────
    # internal helpers
    # ──────────────────────────────────────────────────────────────────────
    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _is_better(self, a, b):
        """Is metric `a` better than `b`?"""
        if b is None:
            return True
        return (a > b) if self.metric_mode == "max" else (a < b)

    def _best_value(self):
        """Current best metric value, or None if no checkpoints yet."""
        if not self.best_ckpts:
            return None
        vals = [m for m, _ in self.best_ckpts]
        return max(vals) if self.metric_mode == "max" else min(vals)

    def _rescan_existing(self):
        """Rebuild the best-checkpoint registry from files already on disk.
        Filenames encode the metric, e.g. epoch_020_val_auc_0.9123.pth"""
        pat = os.path.join(self.dir, f"epoch_*_{self.metric_name}_*.pth")
        for fp in glob.glob(pat):
            try:
                # ..._<metric_name>_<value>.pth
                stem = os.path.basename(fp).rsplit(".pth", 1)[0]
                val = float(stem.split(f"{self.metric_name}_")[-1])
                self.best_ckpts.append((val, fp))
            except Exception:
                continue
        self._sort_best()

    def _sort_best(self):
        """Sort best list so the BEST is first."""
        self.best_ckpts.sort(key=lambda x: x[0],
                             reverse=(self.metric_mode == "max"))

    def _atomic_save(self, state, path):
        """Write to a temp file then rename — never leaves a half-written file."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.dir, suffix=".tmp")
        os.close(tmp_fd)
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, path)        # atomic on POSIX
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _rng_state():
        """Capture all RNG states for exact resume."""
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng(state):
        try:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"])
            if torch.cuda.is_available() and "torch_cuda" in state:
                torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception as e:
            print(f"[ckpt] RNG restore skipped ({e})", flush=True)

    # ──────────────────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────────────────
    def build_state(self, epoch, global_step, metric_value,
                    model=None, optimizer=None, scheduler=None,
                    scaler=None, config=None, extra=None):
        """Assemble the full checkpoint dict."""
        state = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "metric_name": self.metric_name,
            "metric_mode": self.metric_mode,
            "metric_value": float(metric_value) if metric_value is not None else None,
            "best_value": self._best_value(),
            "history": self.history,
            "rng": self._rng_state(),
        }
        if model is not None:
            state["model"] = (model.state_dict() if hasattr(model, "state_dict")
                              else model)
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler"] = scaler.state_dict()
        if config is not None:
            # store config as a plain container so it's portable
            try:
                from omegaconf import OmegaConf
                state["config"] = OmegaConf.to_container(config, resolve=True)
            except Exception:
                state["config"] = config
        if extra:
            state["extra"] = extra
        return state

    def save(self, epoch, global_step, metric_value,
             model=None, optimizer=None, scheduler=None,
             scaler=None, config=None, extra=None):
        """Save latest.pth always; save a ranked checkpoint if it's periodic
        or top-K worthy. Returns the path of any ranked checkpoint saved."""
        state = self.build_state(epoch, global_step, metric_value,
                                 model, optimizer, scheduler, scaler,
                                 config, extra)

        # 1) always write latest.pth (for auto-resume)
        latest_path = os.path.join(self.dir, "latest.pth")
        self._atomic_save(state, latest_path)

        # update history
        if metric_value is not None:
            self.history.append((int(epoch), float(metric_value)))
            state["history"] = self.history

        saved_path = None
        is_periodic = (self.save_every > 0 and
                       (epoch + 1) % self.save_every == 0)
        is_topk = (metric_value is not None and
                   (len(self.best_ckpts) < self.top_k or
                    self._is_better(metric_value,
                                    self.best_ckpts[-1][0])))

        # 2) periodic and/or top-K checkpoint with a descriptive filename
        if metric_value is not None and (is_periodic or is_topk):
            fname = (f"epoch_{epoch:03d}_"
                     f"{self.metric_name}_{metric_value:.4f}.pth")
            saved_path = os.path.join(self.dir, fname)
            self._atomic_save(state, saved_path)
            self._log(f"[ckpt] saved {fname}"
                      f"{'  [periodic]' if is_periodic else ''}"
                      f"{'  [top-k]' if is_topk else ''}")

            # register in best list and prune
            if is_topk:
                self.best_ckpts.append((float(metric_value), saved_path))
                self._sort_best()
                self._prune()

        return saved_path

    def _prune(self):
        """Keep only top_k best checkpoints; delete the rest (but never delete
        a file that's also the latest periodic one if it ranks out — periodic
        files that aren't top-k simply aren't tracked here)."""
        while len(self.best_ckpts) > self.top_k:
            _, worst_path = self.best_ckpts.pop()   # last = worst after sort
            if os.path.exists(worst_path):
                try:
                    os.remove(worst_path)
                    self._log(f"[ckpt] pruned {os.path.basename(worst_path)}")
                except OSError as e:
                    self._log(f"[ckpt] prune failed: {e}")

    def best_path(self):
        """Path to the single best checkpoint, or None."""
        return self.best_ckpts[0][1] if self.best_ckpts else None

    # ──────────────────────────────────────────────────────────────────────
    # resume
    # ──────────────────────────────────────────────────────────────────────
    def resolve_resume_path(self, resume="auto"):
        """Translate a resume directive into a concrete path or None.
            resume == "none"  -> None (fresh start)
            resume == "auto"  -> <dir>/latest.pth if it exists else None
            resume == <path>  -> that path if it exists else None
        """
        if resume is None:
            return None
        resume = str(resume)
        if resume.lower() in ("none", "false", ""):
            return None
        if resume.lower() == "auto":
            latest = os.path.join(self.dir, "latest.pth")
            return latest if os.path.isfile(latest) else None
        # explicit path
        return resume if os.path.isfile(resume) else None

    def maybe_resume(self, resume="auto", map_location="cpu",
                     restore_rng=True):
        """Load a checkpoint per the resume directive. Returns the loaded
        state dict (so the caller can restore model/opt/etc.) or None."""
        path = self.resolve_resume_path(resume)
        if path is None:
            self._log(f"[ckpt] no resume (resume={resume}); starting fresh")
            return None
        self._log(f"[ckpt] resuming from {path}")
        state = torch.load(path, map_location=map_location, weights_only=False)
        # restore history + best registry so top-k logic continues correctly
        self.history = state.get("history", [])
        if restore_rng and "rng" in state:
            self._restore_rng(state["rng"])
        self._log(f"[ckpt] resumed epoch={state.get('epoch')} "
                  f"step={state.get('global_step')} "
                  f"best={state.get('best_value')}")
        return state