#!/usr/bin/env python
"""Pretrain + freeze the artifact probe. Run once before training."""
import argparse, os, sys, torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iganer.utils.config import load_config
from iganer.utils.seed import set_seed
from iganer.utils.metrics import auc
from iganer.data.ffpp import build_dataset
from iganer.probes.artifact_probe import ArtifactProbe

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/default.yaml")
    ap.add_argument("--experiment",default=None); ap.add_argument("overrides",nargs="*")
    a=ap.parse_args(); cfg=load_config(a.config,a.experiment,a.overrides); set_seed(cfg.seed)
    device=cfg.device if torch.cuda.is_available() else "cpu"
    dl=DataLoader(build_dataset(cfg,"train"),batch_size=cfg.train.batch_size,shuffle=True,
                  num_workers=cfg.data.num_workers,drop_last=True)
    probe=ArtifactProbe().to(device); opt=torch.optim.AdamW(probe.parameters(),lr=1e-4,weight_decay=1e-4)
    for ep in range(cfg.probe.pretrain_epochs):
        s,l=[],[]
        for x,y in dl:
            x,y=x.to(device),y.float().to(device); logit=probe.logits(x)
            loss=F.binary_cross_entropy_with_logits(logit,y)
            opt.zero_grad(); loss.backward(); opt.step()
            s+=torch.sigmoid(logit).detach().cpu().tolist(); l+=y.cpu().tolist()
        print(f"epoch {ep}: probe AUC={auc(s,l):.4f}")
    os.makedirs(os.path.dirname(cfg.probe.ckpt) or ".",exist_ok=True)
    torch.save(probe.state_dict(),cfg.probe.ckpt); print(f"saved -> {cfg.probe.ckpt}")
if __name__=="__main__": main()
