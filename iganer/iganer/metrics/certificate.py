"""
iganer/metrics/certificate.py   (NEW)

The headline evaluation of SIGIL: turn "is the identity gap suppressible?" into a
number. We DO NOT claim formal/provable robustness — this is an EMPIRICAL margin
under a specified adversary class. Always report it as such.

Outputs:
  * clean AUC / EER, concealed AUC / EER, robustness gap (clean - concealed)
  * budget-AUC curve: AUC as a function of perceptual budget eps
  * concealment margin: smallest eps at which population AUC drops below a target
    (e.g. 0.70), and per-sample flip thresholds.

A causal, hard-to-suppress identity invariant => large margin (AUC holds until eps
is large enough to violate the identity budget). A shortcut => AUC collapses at
tiny eps. That curve is the falsification test.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import numpy as np
import torch
from torch import Tensor


# ---------- basic metrics (no sklearn dependency) ----------
def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via rank statistic (Mann-Whitney U). labels in {0,1}, 1=fake."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ties
    s = np.concatenate([pos, neg])
    _, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    start = cum - counts
    avg = (start + cum + 1) / 2.0
    ranks[order] = avg[inv]
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def eer(scores: np.ndarray, labels: np.ndarray, n: int = 512) -> float:
    """Equal Error Rate via threshold sweep."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ths = np.linspace(scores.min(), scores.max(), n)
    best, best_gap = 0.5, 1e9
    for t in ths:
        far = (neg >= t).mean()
        frr = (pos < t).mean()
        if abs(far - frr) < best_gap:
            best_gap, best = abs(far - frr), (far + frr) / 2
    return float(best)


@dataclass
class CertificateResult:
    clean_auc: float
    concealed_auc: float
    robustness_gap: float
    clean_eer: float
    concealed_eer: float
    budget_curve: Dict[float, float] = field(default_factory=dict)   # eps -> AUC
    margin_eps: Optional[float] = None        # eps where AUC first < auc_target
    auc_target: float = 0.70

    def as_logs(self, prefix: str = "") -> Dict[str, float]:
        d = {
            f"{prefix}clean_auc": self.clean_auc,
            f"{prefix}concealed_auc": self.concealed_auc,
            f"{prefix}robustness_gap": self.robustness_gap,
            f"{prefix}clean_eer": self.clean_eer,
            f"{prefix}concealed_eer": self.concealed_eer,
            f"{prefix}margin_eps": self.margin_eps if self.margin_eps is not None else -1.0,
        }
        for eps, a in self.budget_curve.items():
            d[f"{prefix}auc@eps={eps:.4f}"] = a
        return d


@torch.no_grad()
def _scores(detector, x: Tensor) -> np.ndarray:
    return torch.sigmoid(detector.fake_logit(x)).detach().cpu().numpy()


def evaluate_certificate(
    detector,
    loader,                       # yields (x, y) or (x, y, x_src); x in [0,1]
    make_concealer: Callable,     # eps -> concealer bound to budget(eps)
    eps_grid: List[float],
    device: str = "cuda",
    auc_target: float = 0.70,
    fixed_eval_eps: Optional[float] = None,   # eps used for the single concealed AUC
) -> CertificateResult:
    """Runs the full budget sweep. `make_concealer(eps)` must return a concealer
    whose Budget.eps_inf == eps (and eps_id/eps_lpips held fixed = the identity and
    perceptual caps that define 'allowed' concealment)."""
    # clean pass
    clean_s, ys = [], []
    cache_x = []
    for batch in loader:
        x, y = batch[0].to(device), batch[1]
        clean_s.append(_scores(detector, x))
        ys.append(np.asarray(y))
        cache_x.append(x.cpu())
    clean_s = np.concatenate(clean_s)
    ys = np.concatenate(ys)
    clean_auc = roc_auc(clean_s, ys)
    clean_eer = eer(clean_s, ys)

    # budget sweep
    curve: Dict[float, float] = {}
    margin = None
    concealed_auc_at_fixed = clean_auc
    for eps in sorted(eps_grid):
        conc = make_concealer(eps)
        s = []
        idx = 0
        for x_cpu in cache_x:
            x = x_cpu.to(device)
            yb = ys[idx: idx + x.size(0)]
            idx += x.size(0)
            fake = torch.tensor(yb == 1, device=device)
            xc = x.clone()
            if fake.any():
                xf = x[fake]
                xc[fake] = conc(detector, xf, suppress_toward_real=True) \
                    if conc.__class__.__name__ == "PGDConcealer" else conc(xf)
            s.append(_scores(detector, xc))
        s = np.concatenate(s)
        a = roc_auc(s, ys)
        curve[eps] = a
        if margin is None and a < auc_target:
            margin = eps
        if fixed_eval_eps is not None and abs(eps - fixed_eval_eps) < 1e-9:
            concealed_auc_at_fixed = a
            concealed_eer_at_fixed = eer(s, ys)

    if fixed_eval_eps is None:
        # default: report the largest eps in the grid as the concealed operating point
        last_eps = sorted(eps_grid)[-1]
        concealed_auc_at_fixed = curve[last_eps]
        # recompute eer at last eps
        conc = make_concealer(last_eps)
        s, idx = [], 0
        for x_cpu in cache_x:
            x = x_cpu.to(device)
            yb = ys[idx: idx + x.size(0)]; idx += x.size(0)
            fake = torch.tensor(yb == 1, device=device)
            xc = x.clone()
            if fake.any():
                xf = x[fake]
                xc[fake] = conc(detector, xf, suppress_toward_real=True) \
                    if conc.__class__.__name__ == "PGDConcealer" else conc(xf)
            s.append(_scores(detector, xc))
        concealed_eer_at_fixed = eer(np.concatenate(s), ys)

    return CertificateResult(
        clean_auc=clean_auc,
        concealed_auc=concealed_auc_at_fixed,
        robustness_gap=clean_auc - concealed_auc_at_fixed,
        clean_eer=clean_eer,
        concealed_eer=concealed_eer_at_fixed,
        budget_curve=curve,
        margin_eps=margin,
        auc_target=auc_target,
    )