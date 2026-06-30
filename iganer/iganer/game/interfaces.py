"""
iganer/game/interfaces.py   (NEW)

Decouples SIGIL from CIFT internals. Your CIFT detector only has to expose this
small surface. The deployed CIFT source-free path already produces a fake-logit;
the training path additionally exposes the donor-target gap embeddings. Wrap your
existing model in a thin adapter that implements `SigilDetector`.

NOTE: keep the detector fully differentiable wrt the *input image* — the inner
maximization (concealment) backpropagates through it.
"""
from __future__ import annotations
from typing import Protocol, Optional, Tuple
import torch
from torch import Tensor


class SigilDetector(Protocol):
    """Minimal contract the CIFT model must satisfy to plug into SIGIL."""

    def fake_logit(self, x: Tensor) -> Tensor:
        """Source-free forward. x:(B,3,H,W) -> (B,) raw logit for P(fake).
        Must use the SAME two-branch source-free graph used at deployment
        (Eq. 9 in CIFT), so the certificate reflects the deployed model."""
        ...

    def gap_embeddings(self, x: Tensor, x_src: Tensor) -> Tuple[Tensor, Tensor]:
        """Training-only privileged path. Returns (g_t, g_s), the XID-Mamba
        identity embeddings for target and donor streams. Δ = ||g_s - g_t||_2.
        Only called during training when a donor reference is available."""
        ...


class IdentityEmbedder(Protocol):
    """Frozen face recognizer used to enforce the identity-preservation budget.
    Plug ArcFace / AdaFace here. Must be frozen (no grad to its params) but
    differentiable wrt input so the constraint can be enforced by projection."""

    def __call__(self, x: Tensor) -> Tensor:  # (B,3,H,W) -> (B,d) L2-normalized
        ...