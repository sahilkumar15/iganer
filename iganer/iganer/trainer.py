"""Factor-aware training loop. ONE function trains ANY of the 8 ablation cells
by reading three booleans from cfg.factors:

  A = cfg.factors.game        learned PPO concealer (True) vs fixed random policy (False)
  B = cfg.factors.targeting   suppression masked to detector saliency (True) vs uniform (False)
  C = cfg.factors.protection  identity-gap protection on reward + detector loss (True) vs off (False)

Both scripts/train.py (single variant) and scripts/run_ablation_tableA.py (all 8)
call train_variant(). This is the modular core of the project.
"""
import os, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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


def _get_lpips(cfg, device):
    if not cfg.reward.use_lpips: return None
    try:
        import lpips; return lpips.LPIPS(net="alex").to(device).eval()
    except Exception as e:
        print(f"[lpips] off ({e}); L1 realism."); return None


def train_variant(cfg, logger, tag="run", probe=None,
                  detector=None, train_loader=None, val_ds=None):
    """Train one ablation cell. Returns (detector, bank, metrics_dict).

    Standalone mode (default): builds the stub detector + synthetic/FF++ data.
    CIFT mode: pass a real CIFTAdapter as `detector` and CIFT's `train_loader`
    + `val_ds` (see train_iganer.py). Each batch must yield (image, label) with
    image in [0,1] BCHW; the adapter handles CIFT's [-1,1] normalization."""
    device = cfg.device if torch.cuda.is_available() else "cpu"
    A = bool(cfg.factors.game)
    B = bool(cfg.factors.targeting)
    C = bool(cfg.factors.protection)

    if val_ds is None:
        val_ds = build_dataset(cfg, "val")
    if train_loader is None:
        train_ds = build_dataset(cfg, "train")
        train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                                  num_workers=cfg.data.num_workers, drop_last=True)
    tl = train_loader

    if detector is None:
        detector = build_detector(cfg, device)
    use_ema  = A and bool(cfg.game.get('use_ema', True))
    det_ema  = EMADetector(detector, cfg.game.ema_decay) if use_ema else None
    reward_target = det_ema if det_ema is not None else detector

    if probe is None:
        probe = ArtifactProbe().to(device)
        if cfg.probe.ckpt and os.path.exists(cfg.probe.ckpt):
            probe.load_state_dict(torch.load(cfg.probe.ckpt, map_location=device))
        probe.freeze()

    bank   = ConcealBank(list(cfg.concealer.operators), cfg.concealer.strength_levels)
    policy = ConcealerPolicy(detector.state_dim, bank.n_actions, cfg.concealer.hidden).to(device) if A else None
    ppo    = PPO(policy, cfg, device) if A else None
    replay = ConcealReplay(cfg.game.replay_cap)
    curr   = Curriculum(cfg)
    lpips_fn = _get_lpips(cfg, device)
    det_opt = torch.optim.AdamW(detector.parameters(), lr=cfg.train.lr,
                                weight_decay=cfg.train.weight_decay)

    step = 0
    for epoch in range(cfg.train.epochs):
        cap = curr.strength_cap(epoch)
        if A: ppo.ent = cfg.concealer.entropy_coef * curr.explore_coef(epoch)
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            fake_idx = torch.where(y == 1)[0]
            x_til_full = x.clone()
            log_extra = {}

            if len(fake_idx) > 0:
                xf = x[fake_idx]
                mask = saliency_mask(detector, xf) if B else None   # factor B

                if A:                                               # factor A: learned
                    with torch.no_grad():
                        state = detector.extract_state(xf)
                    actions, logp, _ = policy.act(state)
                else:                                               # fixed random policy
                    actions = torch.randint(0, bank.n_actions, (xf.shape[0],), device=device)

                x_til = torch.stack([
                    bank.apply(xf[i], int(actions[i]),
                               (mask[i] if mask is not None else None), cap)
                    for i in range(xf.shape[0])])
                x_til_full[fake_idx] = x_til
                replay.add(x_til)

                if A:                                               # PPO update (game)
                    reward, rparts = conceal_reward(reward_target, xf, x_til, cfg,
                                                    protection=C, lpips_fn=lpips_fn)
                    log_extra.update(ppo.update(state, actions, logp, reward))
                    log_extra.update(rparts)

            # ---- Detector update (reals + concealed fakes, + replay) ----
            xb, yb = x_til_full, y
            rep = replay.sample(cfg.train.batch_size // 2, device)
            if rep is not None:
                xb = torch.cat([xb, rep], 0)
                yb = torch.cat([yb, torch.ones(rep.shape[0], device=device)], 0)
            det_loss = detector.compute_loss(xb, yb)

            if C and len(fake_idx) > 0:                             # factor C: detector-side
                gap_clean = detector.identity_gap(x[fake_idx])
                gap_supp  = detector.identity_gap(x_til_full[fake_idx])
                det_loss = det_loss + cfg.reward.delta_protect * F.relu(gap_clean - gap_supp).mean()

            det_opt.zero_grad(); det_loss.backward(); det_opt.step()
            if det_ema is not None: det_ema.update(detector)

            if step % cfg.train.log_every == 0:
                logger.log({"tag": tag, "epoch": epoch, "det_loss": det_loss.item(),
                            "cap": cap, **log_extra}, step)
            step += 1

    v_auc, v_eer = plain_auc(detector, val_ds, cfg, device)
    metrics = {"val_auc": v_auc, "val_eer": v_eer}
    if cfg.train.ckpt_dir:
        os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
        torch.save({"detector": detector.state_dict(),
                    "factors": dict(game=A, targeting=B, protection=C)},
                   os.path.join(cfg.train.ckpt_dir, f"{tag}.pt"))
    return detector, bank, metrics
