"""
iganer/losses/sigil_losses.py   (NEW)

All SIGIL-specific objectives. CIFT's own losses (L_cls focal+BCE, L_diff, IGS
L_IG / L_gap) stay in the CIFT codebase; SIGIL adds the four terms below and reuses
CIFT's L_cls via `classification_loss`.

Outer detector objective (per batch):
    L = L_cls(D(x), y)
      + lam_inv * L_cls(D(x'), y)        # robust term on concealed images
      + lam_sig * L_suppression_invariance(D, x, x')
      + lam_gap * L_idgap_consistency(D, x, x', x_src)   # training-only, privileged
      + (CIFT privileged terms)
where x' = concealer(x) is the in-budget worst-case concealment.
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def classification_loss(logit: Tensor, y: Tensor, gamma: float = 2.0,
                        bce_w: float = 0.5) -> Tensor:
    """Focal + BCE, matching CIFT's L_cls spirit. logit:(B,), y:(B,) in {0,1}."""
    y = y.float()
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    p = torch.sigmoid(logit)
    pt = torch.where(y == 1, p, 1 - p)
    focal = ((1 - pt) ** gamma) * bce
    return (bce_w * bce + (1 - bce_w) * focal).mean()


def suppression_invariance_loss(logit_clean: Tensor, logit_concealed: Tensor,
                                detach_clean: bool = True) -> Tensor:
    """L_sig: the decision must not move under budgeted concealment.
    Penalize the squared change in the fake-logit between x and x'.
    This is the loss form of the 'suppression-invariance certificate'."""
    ref = logit_clean.detach() if detach_clean else logit_clean
    return F.mse_loss(logit_concealed, ref)


def perceptual_preservation_loss(x: Tensor, x_concealed: Tensor,
                                 lpips_fn=None) -> Tensor:
    """Regularizer used when TRAINING an amortized concealer, to keep its outputs
    perceptually faithful. (For PGD this is enforced by projection; kept here so
    both concealers share one budget semantics.)"""
    l1 = F.l1_loss(x_concealed, x)
    if lpips_fn is not None:
        return l1 + lpips_fn(x, x_concealed).mean()
    return l1


def identity_gap_consistency_loss(gap_clean: Tensor, gap_concealed: Tensor) -> Tensor:
    """L_idgap: the donor-target gap Δ = ||g_s - g_t|| must NOT collapse under
    concealment for forged pairs. Binds the adversary to CIFT's actual mechanism:
    if the concealer can shrink Δ within budget, the gap was fragile.
    gap_*:(B,) are Δ computed on (x, x_src) and (x', x_src). Forged-only."""
    return F.l1_loss(gap_concealed, gap_clean.detach())


def gap_from_embeddings(g_t: Tensor, g_s: Tensor) -> Tensor:
    """Δ = ||g_s - g_t||_2 per sample."""
    return (g_s - g_t).norm(dim=-1)


class SigilLossWeights:
    def __init__(self, lam_inv=1.0, lam_sig=0.5, lam_gap=0.5,
                 lam_cls=1.0, warmup_epochs=3):
        self.lam_inv = lam_inv
        self.lam_sig = lam_sig
        self.lam_gap = lam_gap
        self.lam_cls = lam_cls
        self.warmup_epochs = warmup_epochs

    def schedule(self, epoch: int) -> "SigilLossWeights":
        """Ramp the robust/invariance terms after a clean warmup so the detector
        and (if used) amortized concealer start from a stable point — analogous to
        TSRL's Student-warmup rationale, but here it stabilizes the min-max game."""
        if epoch < self.warmup_epochs:
            f = epoch / max(1, self.warmup_epochs)
            return SigilLossWeights(self.lam_inv * f, self.lam_sig * f,
                                    self.lam_gap * f, self.lam_cls,
                                    self.warmup_epochs)
        return self