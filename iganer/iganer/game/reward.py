"""Concealer reward (factor C gates the protection term beta):
  r = a*[D(x)-D(x_til)]_+ - b*[Δ(x)-Δ(x_til)]_+ - c*realism
b is zeroed when protection is OFF."""
import torch
@torch.no_grad()
def conceal_reward(det_ema, x, x_til, cfg, protection=True, lpips_fn=None):
    a=cfg.reward.alpha; b=(cfg.reward.beta if protection else 0.0); c=cfg.reward.gamma
    conceal=(det_ema.fake_prob(x)-det_ema.fake_prob(x_til)).clamp(min=0)
    gap_pen=(det_ema.identity_gap(x)-det_ema.identity_gap(x_til)).clamp(min=0)
    realism=(lpips_fn(x*2-1,x_til*2-1).flatten() if lpips_fn is not None
             else (x-x_til).abs().flatten(1).mean(1))
    r=a*conceal-b*gap_pen-c*realism
    return r, dict(r_conceal=conceal.mean().item(), r_gap_pen=gap_pen.mean().item(),
                   r_realism=realism.mean().item(), reward=r.mean().item())
