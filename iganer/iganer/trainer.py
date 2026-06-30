"""Factor-aware training loop — memory-optimised for large CIFT model.

Key fixes vs original:
  1. gradient_checkpointing on the CIFT encoder to cut activation memory ~40%
  2. torch.cuda.empty_cache() after concealer ops (before detector backward)
  3. identity_gap calls wrapped in torch.no_grad() where gradients not needed
  4. tqdm progress bar + flushed console heartbeat (visible even with W&B on)
  5. cfg.train.max_steps cap for fast smoke tests
  6. cfg.train.grad_clip_val respected (already in config, was ignored before)
"""
import os, sys, time, torch
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
from .eval import plain_auc


def _hb(msg):
    """Heartbeat: always prints to console, immediately flushed."""
    print(msg, flush=True)
    sys.stdout.flush()


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
    """Enable gradient checkpointing on the CIFT encoder to save ~40% activation memory."""
    try:
        # CIFTAdapter: detector.gn is GuideNet, detector.gn.encoder is the ConvNeXt
        enc = getattr(getattr(detector, 'gn', None), 'encoder', None)
        if enc is not None and hasattr(enc, 'set_grad_checkpointing'):
            enc.set_grad_checkpointing(enable=True)
            _hb("[memory] gradient checkpointing ON for GuideNet encoder")
            return True
        # fallback: try timm's standard interface
        if enc is not None and hasattr(enc, 'grad_checkpointing'):
            enc.grad_checkpointing = True
            _hb("[memory] gradient checkpointing ON (direct attr)")
            return True
    except Exception as e:
        _hb(f"[memory] grad checkpoint unavailable: {e}")
    return False


def train_variant(cfg, logger, tag="run", probe=None,
                  detector=None, train_loader=None, val_ds=None):
    """Train one ablation cell. Returns (detector, bank, metrics_dict)."""
    device = cfg.device if torch.cuda.is_available() else "cpu"
    A = bool(cfg.factors.game)
    B = bool(cfg.factors.targeting)
    C = bool(cfg.factors.protection)

    max_steps  = int(cfg.train.get("max_steps", 0) or 0)
    log_every  = int(cfg.train.get("log_every", 50))
    grad_clip  = float(cfg.train.get("grad_clip_val", 0.5))

    _hb(f"[{tag}] starting  A={A} B={B} C={C}  device={device}  "
        f"max_steps={max_steps or 'full'}  grad_clip={grad_clip}")

    # ── data ──────────────────────────────────────────────────────────────────
    if val_ds is None:
        val_ds = build_dataset(cfg, "val")
    if train_loader is None:
        train_ds = build_dataset(cfg, "train")
        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=True,
            num_workers=cfg.data.num_workers, drop_last=True,
            pin_memory=True, persistent_workers=(cfg.data.num_workers > 0))

    # ── detector ──────────────────────────────────────────────────────────────
    if detector is None:
        detector = build_detector(cfg, device)

    # enable gradient checkpointing to save activation memory
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

    # ── concealer + optimiser ─────────────────────────────────────────────────
    bank   = ConcealBank(list(cfg.concealer.operators), cfg.concealer.strength_levels)
    policy = (ConcealerPolicy(detector.state_dim, bank.n_actions,
                              cfg.concealer.hidden).to(device) if A else None)
    ppo    = PPO(policy, cfg, device) if A else None
    replay = ConcealReplay(cfg.game.replay_cap)
    curr   = Curriculum(cfg)
    lpips_fn = _get_lpips(cfg, device)

    det_opt = torch.optim.AdamW(
        detector.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay)

    _hb(f"[{tag}] ready — n_actions={bank.n_actions} "
        f"batch={cfg.train.batch_size} workers={cfg.data.num_workers}")

    # ── training loop ─────────────────────────────────────────────────────────
    step   = 0
    t_last = time.time()
    stop   = False

    for epoch in range(cfg.train.epochs):
        if stop:
            break
        cap = curr.strength_cap(epoch)
        if A:
            ppo.ent = cfg.concealer.entropy_coef * curr.explore_coef(epoch)

        pbar = tqdm(train_loader,
                    desc=f"[{tag}] ep{epoch}",
                    dynamic_ncols=True, leave=False)

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            fake_idx   = torch.where(y == 1)[0]
            x_til_full = x.clone()
            log_extra  = {}

            # ── concealer step (no grad needed here) ──────────────────────
            if len(fake_idx) > 0:
                xf = x[fake_idx]

                # factor B: saliency mask (needs one backward; keep separate)
                mask = saliency_mask(detector, xf) if B else None

                if A:
                    with torch.no_grad():
                        state = detector.extract_state(xf)
                    actions, logp, _ = policy.act(state)
                else:
                    actions = torch.randint(
                        0, bank.n_actions, (xf.shape[0],), device=device)

                # apply suppression operators (CPU-side, no GPU memory held)
                x_til = torch.stack([
                    bank.apply(xf[i], int(actions[i]),
                               (mask[i] if mask is not None else None), cap)
                    for i in range(xf.shape[0])])
                x_til_full[fake_idx] = x_til
                replay.add(x_til)

                # PPO reward + update (factor A)
                if A:
                    with torch.no_grad():
                        reward, rparts = conceal_reward(
                            reward_target, xf, x_til, cfg,
                            protection=C, lpips_fn=lpips_fn)
                    log_extra.update(
                        ppo.update(state, actions, logp, reward))
                    log_extra.update(rparts)

            # free any lingering cached memory before the big backward
            torch.cuda.empty_cache()

            # ── detector update ───────────────────────────────────────────
            xb, yb = x_til_full, y
            rep = replay.sample(cfg.train.batch_size // 2, device)
            if rep is not None:
                xb = torch.cat([xb, rep], 0)
                yb = torch.cat(
                    [yb, torch.ones(rep.shape[0], device=device)], 0)

            det_loss = detector.compute_loss(xb, yb)

            # factor C: identity-gap regulariser on detector side
            if C and len(fake_idx) > 0:
                with torch.no_grad():
                    gap_clean = detector.identity_gap(x[fake_idx])
                gap_supp = detector.identity_gap(x_til_full[fake_idx])
                det_loss = (det_loss +
                            cfg.reward.delta_protect *
                            F.relu(gap_clean - gap_supp).mean())

            det_opt.zero_grad()
            det_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    detector.parameters(), grad_clip)
            det_opt.step()

            if det_ema is not None:
                det_ema.update(detector)

            # ── logging ───────────────────────────────────────────────────
            loss_val = det_loss.item()
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    loss=f"{loss_val:.4f}", cap=f"{cap:.2f}")

            if step % log_every == 0:
                dt  = time.time() - t_last
                ips = (log_every / dt) if (step > 0 and dt > 0) else 0.0
                _hb(f"[{tag}] step={step} epoch={epoch} "
                    f"det_loss={loss_val:.4f} cap={cap:.3f} "
                    f"{ips:.2f} it/s")
                t_last = time.time()
                logger.log(
                    {"tag": tag, "epoch": epoch,
                     "det_loss": loss_val, "cap": cap,
                     **log_extra}, step)

            step += 1
            if max_steps and step >= max_steps:
                _hb(f"[{tag}] hit max_steps={max_steps}; stopping early")
                stop = True
                break

    # ── validation ────────────────────────────────────────────────────────────
    _hb(f"[{tag}] training done ({step} steps); computing val AUC ...")
    v_auc, v_eer = plain_auc(detector, val_ds, cfg, device)
    _hb(f"[{tag}] val_auc={v_auc:.4f}  val_eer={v_eer:.4f}")

    metrics = {"val_auc": v_auc, "val_eer": v_eer}
    if cfg.train.ckpt_dir:
        os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
        out = os.path.join(cfg.train.ckpt_dir, f"{tag}.pt")
        torch.save({"detector": detector.state_dict(),
                    "factors":  dict(game=A, targeting=B, protection=C)}, out)
        _hb(f"[{tag}] checkpoint -> {out}")

    return detector, bank, metrics