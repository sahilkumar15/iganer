"""Factor-aware training loop with robust checkpointing, resume, AMP, scheduler,
per-epoch validation, and rich W&B logging.

ONE function trains ANY of the 8 ablation cells by reading three booleans:
  A = cfg.factors.game        learned PPO concealer vs fixed random policy
  B = cfg.factors.targeting   suppression masked to saliency vs uniform
  C = cfg.factors.protection  identity-gap protection on reward + detector

New (vs original):
  - per-epoch validation (AUC, EER, loss, accuracy)
  - CheckpointManager: periodic + top-K best + latest.pth, per-variant dir
  - auto/path/none resume that restores model+opt+sched+scaler+epoch+step+rng
  - optional mixed precision (AMP) via cfg.amp.enabled
  - cosine-warmup scheduler from cfg.train.scheduler
  - gradient checkpointing + empty_cache for memory
  - clean per-epoch W&B + console logging ("Epoch X/Y")
"""
import os, sys, time, math, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

from .data.ffpp import build_dataset
from .detector.interface import build_detector
from .probes.artifact_probe import ArtifactProbe
from .concealer.operators import ConcealBank
from .concealer.policy import ConcealerPolicy
from .concealer.attribution import saliency_mask
from .game.reward import conceal_reward
from .game.ppo import PPO
from .game.ema import EMADetector
from .game.replay import ConcealReplay
from .game.curriculum import Curriculum
from .eval import plain_auc, validate
from .utils.checkpoint import CheckpointManager


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _hb(msg):
    print(msg, flush=True); sys.stdout.flush()


def _get_lpips(cfg, device):
    if not cfg.reward.use_lpips:
        return None
    try:
        import lpips
        return lpips.LPIPS(net="alex").to(device).eval()
    except Exception as e:
        _hb(f"[lpips] off ({e}); using L1 realism.")
        return None


def _enable_grad_checkpointing(detector):
    try:
        enc = getattr(getattr(detector, 'gn', None), 'encoder', None)
        if enc is not None and hasattr(enc, 'set_grad_checkpointing'):
            enc.set_grad_checkpointing(enable=True)
            _hb("[memory] gradient checkpointing ON for GuideNet encoder")
    except Exception as e:
        _hb(f"[memory] grad checkpoint unavailable: {e}")


def _build_scheduler(optimizer, cfg, steps_per_epoch):
    """Cosine schedule with linear warmup, read from cfg.train.scheduler.
    Returns (scheduler, step_interval) where interval is 'step' or 'epoch'."""
    sch_cfg = cfg.train.get("scheduler", None)
    if not sch_cfg:
        return None, "epoch"
    name = sch_cfg.get("name", "cosine_warmup")
    interval = sch_cfg.get("interval", "step")
    total_epochs = int(cfg.train.epochs)

    if name == "cosine_warmup":
        cw = sch_cfg.get("cosine_warmup", {})
        warmup_steps = int(cw.get("warmup_steps", 0))
        min_lr_factor = float(cw.get("min_lr_factor", 0.05))
        total_steps = (steps_per_epoch * total_epochs
                       if interval == "step" else total_epochs)
        total_steps = max(total_steps, 1)

        def lr_lambda(cur):
            if warmup_steps > 0 and cur < warmup_steps:
                return cur / max(warmup_steps, 1)
            prog = (cur - warmup_steps) / max(total_steps - warmup_steps, 1)
            prog = min(max(prog, 0.0), 1.0)
            cos = 0.5 * (1 + math.cos(math.pi * prog))
            return min_lr_factor + (1 - min_lr_factor) * cos

        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return sched, interval
    return None, "epoch"


# ──────────────────────────────────────────────────────────────────────────────
# main training entry
# ──────────────────────────────────────────────────────────────────────────────
def train_variant(cfg, logger, tag="run", probe=None,
                  detector=None, train_loader=None, val_ds=None):
    """Train one ablation cell. Returns (detector, bank, metrics_dict)."""
    device = cfg.device if torch.cuda.is_available() else "cpu"
    A = bool(cfg.factors.game)
    B = bool(cfg.factors.targeting)
    C = bool(cfg.factors.protection)

    total_epochs = int(cfg.train.epochs)
    max_steps    = int(cfg.train.get("max_steps", 0) or 0)
    log_every    = int(cfg.train.get("log_every", 50))
    grad_clip    = float(cfg.train.get("grad_clip_val", 0.5))

    # ── checkpoint config (generalized, from YAML) ────────────────────────────
    ck_cfg       = cfg.get("checkpoint", {}) or {}
    base_ck_dir  = ck_cfg.get("dir", cfg.train.get("ckpt_dir", "outputs/ckpt"))
    ckpt_dir     = os.path.join(base_ck_dir, tag)          # per-variant subdir
    metric_name  = ck_cfg.get("metric", "val_auc")
    metric_mode  = ck_cfg.get("mode", "max")
    save_every   = int(ck_cfg.get("save_every", 10))
    top_k        = int(ck_cfg.get("top_k", 3))
    resume_dir   = str(cfg.get("resume", "auto"))

    # ── AMP config ────────────────────────────────────────────────────────────
    amp_cfg     = cfg.get("amp", {}) or {}
    use_amp     = bool(amp_cfg.get("enabled", False)) and device.startswith("cuda")

    _hb(f"[{tag}] A={A} B={B} C={C} | epochs={total_epochs} "
        f"batch={cfg.train.batch_size} amp={use_amp} | ckpt_dir={ckpt_dir}")

    # ── data ──────────────────────────────────────────────────────────────────
    if val_ds is None:
        val_ds = build_dataset(cfg, "val")
    if train_loader is None:
        train_ds = build_dataset(cfg, "train")
        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, drop_last=True,
            pin_memory=True, persistent_workers=(cfg.data.num_workers > 0))
    steps_per_epoch = max(len(train_loader), 1)

    # ── detector ──────────────────────────────────────────────────────────────
    if detector is None:
        detector = build_detector(cfg, device)
    _enable_grad_checkpointing(detector)

    use_ema       = A and bool(cfg.game.get('use_ema', True))
    det_ema       = EMADetector(detector, cfg.game.ema_decay) if use_ema else None
    reward_target = det_ema if det_ema is not None else detector

    # ── probe ─────────────────────────────────────────────────────────────────
    if probe is None:
        probe = ArtifactProbe().to(device)
        if cfg.probe.ckpt and os.path.exists(cfg.probe.ckpt):
            probe.load_state_dict(torch.load(cfg.probe.ckpt, map_location=device))
        probe.freeze()

    # ── concealer + optim + sched + scaler ────────────────────────────────────
    bank   = ConcealBank(list(cfg.concealer.operators), cfg.concealer.strength_levels)
    policy = (ConcealerPolicy(detector.state_dim, bank.n_actions,
                              cfg.concealer.hidden).to(device) if A else None)
    ppo    = PPO(policy, cfg, device) if A else None
    replay = ConcealReplay(cfg.game.replay_cap)
    curr   = Curriculum(cfg)
    lpips_fn = _get_lpips(cfg, device)

    det_opt = torch.optim.AdamW(detector.parameters(), lr=cfg.train.lr,
                                weight_decay=cfg.train.weight_decay)
    scheduler, sch_interval = _build_scheduler(det_opt, cfg, steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── checkpoint manager + resume ───────────────────────────────────────────
    ckpt_mgr = CheckpointManager(ckpt_dir, metric_name=metric_name,
                                 metric_mode=metric_mode,
                                 save_every=save_every, top_k=top_k)

    start_epoch = 0
    global_step = 0
    state = ckpt_mgr.maybe_resume(resume=resume_dir, map_location=device)
    if state is not None:
        try:
            detector.load_state_dict(state["model"], strict=False)
            if "optimizer" in state: det_opt.load_state_dict(state["optimizer"])
            if scheduler is not None and "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            if use_amp and "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            start_epoch = int(state.get("epoch", -1)) + 1
            global_step = int(state.get("global_step", 0))
            _hb(f"[{tag}] resumed -> start_epoch={start_epoch} step={global_step}")
        except Exception as e:
            _hb(f"[{tag}] resume partial ({e}); continuing from epoch {start_epoch}")

    _hb(f"[{tag}] ready — n_actions={bank.n_actions} steps/epoch={steps_per_epoch} "
        f"resume='{resume_dir}'")

    # ──────────────────────────────────────────────────────────────────────────
    # training loop
    # ──────────────────────────────────────────────────────────────────────────
    stop = False
    last_train_loss = float("nan")
    best_metric = ckpt_mgr._best_value()

    for epoch in range(start_epoch, total_epochs):
        if stop:
            break
        detector.train()
        cap = curr.strength_cap(epoch)
        if A:
            ppo.ent = cfg.concealer.entropy_coef * curr.explore_coef(epoch)

        epoch_loss_sum, epoch_loss_n = 0.0, 0
        t_last = time.time()
        pbar = tqdm(train_loader, desc=f"[{tag}] ep{epoch}/{total_epochs}",
                    dynamic_ncols=True, leave=False)

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            fake_idx   = torch.where(y == 1)[0]
            x_til_full = x.clone()
            log_extra  = {}

            # ── concealer step ────────────────────────────────────────────
            if len(fake_idx) > 0:
                xf = x[fake_idx]
                mask = saliency_mask(detector, xf) if B else None

                if A:
                    with torch.no_grad():
                        state_feat = detector.extract_state(xf)
                    actions, logp, _ = policy.act(state_feat)
                else:
                    actions = torch.randint(0, bank.n_actions,
                                            (xf.shape[0],), device=device)

                x_til = torch.stack([
                    bank.apply(xf[i], int(actions[i]),
                               (mask[i] if mask is not None else None), cap)
                    for i in range(xf.shape[0])])
                x_til_full[fake_idx] = x_til
                replay.add(x_til)

                if A:
                    with torch.no_grad():
                        reward, rparts = conceal_reward(
                            reward_target, xf, x_til, cfg,
                            protection=C, lpips_fn=lpips_fn)
                    log_extra.update(ppo.update(state_feat, actions, logp, reward))
                    log_extra.update(rparts)

            torch.cuda.empty_cache()

            # ── detector update (with optional AMP) ───────────────────────
            xb, yb = x_til_full, y
            rep = replay.sample(cfg.train.batch_size // 2, device)
            if rep is not None:
                xb = torch.cat([xb, rep], 0)
                yb = torch.cat([yb, torch.ones(rep.shape[0], device=device)], 0)

            det_opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                det_loss = detector.compute_loss(xb, yb)
                if C and len(fake_idx) > 0:
                    with torch.no_grad():
                        gap_clean = detector.identity_gap(x[fake_idx])
                    gap_supp = detector.identity_gap(x_til_full[fake_idx])
                    det_loss = det_loss + cfg.reward.delta_protect * \
                        F.relu(gap_clean - gap_supp).mean()

            scaler.scale(det_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(det_opt)
                torch.nn.utils.clip_grad_norm_(detector.parameters(), grad_clip)
            scaler.step(det_opt)
            scaler.update()

            if scheduler is not None and sch_interval == "step":
                scheduler.step()
            if det_ema is not None:
                det_ema.update(detector)

            # ── bookkeeping ───────────────────────────────────────────────
            loss_val = float(det_loss.item())
            epoch_loss_sum += loss_val; epoch_loss_n += 1
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{loss_val:.4f}", cap=f"{cap:.2f}")

            if global_step % log_every == 0:
                dt  = time.time() - t_last
                ips = (log_every / dt) if dt > 0 else 0.0
                cur_lr = det_opt.param_groups[0]["lr"]
                _hb(f"[{tag}] step={global_step} ep={epoch}/{total_epochs} "
                    f"loss={loss_val:.4f} lr={cur_lr:.2e} cap={cap:.3f} {ips:.2f}it/s")
                logger.log({"tag": tag, "train_loss_step": loss_val,
                            "lr": cur_lr, "cap": cap, **log_extra}, global_step)
                t_last = time.time()

            global_step += 1
            if max_steps and global_step >= max_steps:
                _hb(f"[{tag}] hit max_steps={max_steps}; stopping")
                stop = True
                break

        # epoch-end scheduler step (if epoch interval)
        if scheduler is not None and sch_interval == "epoch":
            scheduler.step()

        last_train_loss = (epoch_loss_sum / epoch_loss_n) if epoch_loss_n else float("nan")

        # ── per-epoch validation ──────────────────────────────────────────
        val_metrics = validate(detector, val_ds, cfg, device)
        metric_value = val_metrics.get(metric_name, None)

        # update best tracking
        if metric_value is not None:
            if best_metric is None or ckpt_mgr._is_better(metric_value, best_metric):
                best_metric = metric_value

        # ── checkpoint (latest always; periodic + top-K as configured) ────
        saved_path = ckpt_mgr.save(
            epoch, global_step, metric_value,
            model=detector, optimizer=det_opt,
            scheduler=scheduler, scaler=(scaler if use_amp else None),
            config=cfg,
            extra={"factors": dict(game=A, targeting=B, protection=C)})

        # ── rich epoch logging ────────────────────────────────────────────
        logger.log_epoch(
            epoch + 1, total_epochs, val_metrics, tag=tag,
            best_metric=best_metric, best_name=metric_name,
            lr=det_opt.param_groups[0]["lr"],
            ckpt_path=saved_path, train_loss=last_train_loss)

    # ── final metrics ──────────────────────────────────────────────────────────
    final = validate(detector, val_ds, cfg, device)
    _hb(f"[{tag}] DONE val_auc={final.get('val_auc', float('nan')):.4f} "
        f"val_eer={final.get('val_eer', float('nan')):.4f} "
        f"best_{metric_name}={best_metric}")

    metrics = {"val_auc": final.get("val_auc"), "val_eer": final.get("val_eer"),
               "best_metric": best_metric}
    return detector, bank, metrics