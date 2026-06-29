"""Suppression-invariance curve: AUC vs artifact-suppression strength.
Δ-drop = AUC(s=0) - AUC(s=1). Flat curve (small Δ-drop) = the detector relies
on the invariant signal, not artifacts. Used by both the eval script and the
ablation runner."""
import numpy as np, torch
from torch.utils.data import DataLoader
from .utils.metrics import auc, eer

@torch.no_grad()
def _auc_at(detector, loader, bank, op_idx, s, device):
    sc,lb=[],[]
    for x,y in loader:
        x=x.to(device); fake=y==1; xs=x.clone()
        if fake.any(): xs[fake]=bank.apply_uniform(x[fake],op_idx,s)
        sc+=detector.fake_prob(xs).cpu().tolist(); lb+=y.tolist()
    return auc(sc,lb), eer(sc,lb)

@torch.no_grad()
def suppression_curve(detector, dataset, bank, cfg, device, points=11, operator=None):
    detector.eval()
    loader=DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                      num_workers=cfg.data.num_workers)
    op_idx=bank.names.index(operator or cfg.eval.curve_operator)
    rows=[]
    for s in np.linspace(0,1,points):
        a,e=_auc_at(detector,loader,bank,op_idx,float(s),device)
        rows.append((float(s),a,e))
    delta_drop=rows[0][1]-rows[-1][1]
    return rows, delta_drop

@torch.no_grad()
def plain_auc(detector, dataset, cfg, device):
    detector.eval()
    loader=DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                      num_workers=cfg.data.num_workers)
    sc,lb=[],[]
    for x,y in loader:
        sc+=detector.fake_prob(x.to(device)).cpu().tolist(); lb+=y.tolist()
    return auc(sc,lb), eer(sc,lb)
