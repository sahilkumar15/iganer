"""Suppression-invariance curve + validation metrics.

- suppression_curve(): AUC vs artifact-suppression strength; Δ-drop = AUC(s=0)-AUC(s=1)
- plain_auc(): quick (AUC, EER) on a dataset (kept for backward compatibility)
- validate(): full per-epoch validation -> {val_loss, val_auc, val_eer, val_accuracy}
"""
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .utils.metrics import auc, eer


@torch.no_grad()
def _auc_at(detector, loader, bank, op_idx, s, device):
    sc, lb = [], []
    for x, y in loader:
        x = x.to(device); fake = y == 1; xs = x.clone()
        if fake.any():
            xs[fake] = bank.apply_uniform(x[fake], op_idx, s)
        sc += detector.fake_prob(xs).cpu().tolist(); lb += y.tolist()
    return auc(sc, lb), eer(sc, lb)


@torch.no_grad()
def suppression_curve(detector, dataset, bank, cfg, device, points=11, operator=None):
    detector.eval()
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers)
    op_idx = bank.names.index(operator or cfg.eval.curve_operator)
    rows = []
    for s in np.linspace(0, 1, points):
        a, e = _auc_at(detector, loader, bank, op_idx, float(s), device)
        rows.append((float(s), a, e))
    delta_drop = rows[0][1] - rows[-1][1]
    return rows, delta_drop


@torch.no_grad()
def plain_auc(detector, dataset, cfg, device):
    """Backward-compatible: returns (auc, eer)."""
    detector.eval()
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers)
    sc, lb = [], []
    for x, y in loader:
        sc += detector.fake_prob(x.to(device)).cpu().tolist(); lb += y.tolist()
    return auc(sc, lb), eer(sc, lb)


@torch.no_grad()
def validate(detector, dataset, cfg, device, max_batches=None):
    """Full validation pass. Returns a metrics dict:
        {val_loss, val_auc, val_eer, val_accuracy}

    val_loss   : mean BCE over the validation set (probe-free, detector logits)
    val_auc    : ROC-AUC of fake_prob vs label
    val_eer    : equal error rate
    val_accuracy: thresholded at 0.5
    """
    detector.eval()
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers)

    scores, labels = [], []
    loss_sum, n = 0.0, 0
    correct, total = 0, 0

    for bi, (x, y) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x = x.to(device)
        yf = y.float().to(device)

        # logits + probability
        logit = detector.logits(x).view(-1)
        prob = torch.sigmoid(logit)

        # loss (plain BCE; robust even if detector.compute_loss is heavy)
        loss = F.binary_cross_entropy_with_logits(
            logit.clamp(-20, 20), yf.view(-1), reduction="sum")
        loss_sum += float(loss.item()); n += yf.numel()

        # accuracy @ 0.5
        pred = (prob > 0.5).long().cpu()
        correct += int((pred == y.long()).sum()); total += y.numel()

        scores += prob.cpu().tolist(); labels += y.tolist()

    return {
        "val_loss":     (loss_sum / max(n, 1)),
        "val_auc":      auc(scores, labels),
        "val_eer":      eer(scores, labels),
        "val_accuracy": (correct / max(total, 1)),
    }