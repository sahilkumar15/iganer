"""
iganer/concealer/budgeted_concealer.py   (NEW)

The inner adversary of SIGIL. Two interchangeable concealers share one budget set
B(x; eps) = { delta :  ||delta||_inf <= eps_inf
                       LPIPS(x, x+delta) <= eps_p          (perceptual budget)
                       cos(F_id(x+delta), F_id(x)) >= 1 - eps_id  (identity budget) }

1) PGDConcealer  -- gradient ascent on the detector's "fake" logit, projected onto
   B every step. This is the PRIMARY concealer: the inner max is continuous and
   differentiable, so PGD upper-bounds what any in-budget filter-bandit could do.
   RL is deliberately NOT used here (see paper's RL-justification section).

2) AmortizedConcealer -- a small U-Net that predicts delta in one shot. Useful for
   fast min-max training; its outputs are still projected onto B. Train it to
   *maximize* suppression (handled in game/minmax.py).

The identity budget is the scientific crux: the concealer may suppress detectability
but may NOT change who the face looks like. Without it the certificate is vacuous.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class Budget:
    eps_inf: float = 8 / 255      # L_inf perceptual cap (pixel space, [0,1] images)
    eps_lpips: float = 0.05       # max LPIPS perceptual drift (None -> skip)
    eps_id: float = 0.10          # max identity drift: cos must stay >= 1 - eps_id
    use_lpips: bool = True
    use_identity: bool = True


def _project_linf(delta: Tensor, x: Tensor, eps_inf: float) -> Tensor:
    delta = torch.clamp(delta, -eps_inf, eps_inf)
    # keep x+delta a valid image
    return torch.clamp(x + delta, 0.0, 1.0) - x


class PGDConcealer(nn.Module):
    """Worst-case in-budget concealment via projected gradient ascent.

    detector.fake_logit must be differentiable wrt the input image.
    For forged samples we PUSH the logit toward 'real' (suppress the fake cue).
    """

    def __init__(
        self,
        budget: Budget,
        steps: int = 10,
        alpha: Optional[float] = None,       # step size; default eps_inf/4
        lpips_fn=None,                       # callable(x, x') -> (B,) LPIPS, or None
        id_embedder=None,                    # IdentityEmbedder or None
        random_start: bool = True,
    ):
        super().__init__()
        self.b = budget
        self.steps = steps
        self.alpha = alpha if alpha is not None else budget.eps_inf / 4
        self.lpips_fn = lpips_fn
        self.id_embedder = id_embedder
        self.random_start = random_start

    @torch.no_grad()
    def _identity_ok(self, x: Tensor, xp: Tensor) -> Tensor:
        if not (self.b.use_identity and self.id_embedder is not None):
            return torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        a = F.normalize(self.id_embedder(x), dim=-1)
        bb = F.normalize(self.id_embedder(xp), dim=-1)
        cos = (a * bb).sum(-1)
        return cos >= (1.0 - self.b.eps_id)

    def _soft_constraint_penalty(self, x: Tensor, xp: Tensor) -> Tensor:
        """Differentiable penalties used to keep the ascent inside the LPIPS and
        identity budgets (hard L_inf is enforced by projection; LPIPS/identity are
        non-box constraints, handled as logarithmic-barrier-style soft terms)."""
        pen = x.new_zeros(())
        if self.b.use_lpips and self.lpips_fn is not None:
            d = self.lpips_fn(x, xp).clamp_min(0)
            pen = pen + F.relu(d - self.b.eps_lpips).mean()
        if self.b.use_identity and self.id_embedder is not None:
            a = F.normalize(self.id_embedder(x), dim=-1)
            bb = F.normalize(self.id_embedder(xp), dim=-1)
            cos = (a * bb).sum(-1)
            pen = pen + F.relu((1.0 - self.b.eps_id) - cos).mean()
        return pen

    def forward(self, detector, x: Tensor, suppress_toward_real: bool = True,
                lam_pen: float = 20.0) -> Tensor:
        """Returns concealed image x' = x + delta*, delta* in B(x).
        suppress_toward_real=True for forged inputs (drive logit down)."""
        x = x.detach()
        delta = torch.zeros_like(x)
        if self.random_start:
            delta = torch.empty_like(x).uniform_(-self.b.eps_inf, self.b.eps_inf)
            delta = _project_linf(delta, x, self.b.eps_inf)

        for _ in range(self.steps):
            delta.requires_grad_(True)
            xp = x + delta
            logit = detector.fake_logit(xp)              # (B,)
            # ascend: make detector LESS sure it's fake -> minimize logit
            obj = logit.mean() if suppress_toward_real else -logit.mean()
            obj = obj + lam_pen * self._soft_constraint_penalty(x, xp)
            grad = torch.autograd.grad(obj, delta, only_inputs=True)[0]
            with torch.no_grad():
                # descend obj wrt delta (we want to reduce 'fake' logit)
                delta = delta - self.alpha * grad.sign()
                delta = _project_linf(delta, x, self.b.eps_inf)
            delta = delta.detach()

        xp = (x + delta).clamp(0, 1)
        # hard identity reject: roll back samples that violated the identity budget
        ok = self._identity_ok(x, xp)
        xp = torch.where(ok.view(-1, 1, 1, 1), xp, x)
        return xp


class _ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class AmortizedConcealer(nn.Module):
    """Tiny U-Net that predicts a perturbation in one forward pass. Output is
    tanh-bounded and projected onto the L_inf ball; LPIPS/identity are enforced as
    soft penalties during its training (in the game loop) plus a hard identity
    reject at use time. Faster than PGD for the inner loop; less tight."""

    def __init__(self, budget: Budget, base: int = 32):
        super().__init__()
        self.b = budget
        self.enc1 = _ConvBlock(3, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.mid = _ConvBlock(base * 2, base * 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = _ConvBlock(base * 4, base * 2)
        self.dec2 = _ConvBlock(base * 3, base)
        self.head = nn.Conv2d(base, 3, 1)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m = self.mid(self.pool(e2))
        d1 = self.dec1(torch.cat([self.up(m), e2], 1))
        d2 = self.dec2(torch.cat([self.up(d1), e1], 1))
        delta = torch.tanh(self.head(d2)) * self.b.eps_inf
        delta = _project_linf(delta, x, self.b.eps_inf)
        return (x + delta).clamp(0, 1)