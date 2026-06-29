#!/usr/bin/env python
"""Standalone suppression curve for one checkpoint (AUC vs strength)."""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iganer.utils.config import load_config
from iganer.utils.logging import Logger
from iganer.detector.interface import build_detector
from iganer.concealer.operators import ConcealBank
from iganer.data.ffpp import build_dataset
from iganer.eval import suppression_curve

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/default.yaml")
    ap.add_argument("--experiment",default=None); ap.add_argument("--ckpt",default=None)
    ap.add_argument("--operator",default=None); ap.add_argument("--points",type=int,default=11)
    ap.add_argument("overrides",nargs="*"); a=ap.parse_args()
    cfg=load_config(a.config,a.experiment,a.overrides)
    device=cfg.device if torch.cuda.is_available() else "cpu"; logger=Logger(cfg)
    det=build_detector(cfg,device)
    if a.ckpt and os.path.exists(a.ckpt):
        det.load_state_dict(torch.load(a.ckpt,map_location=device,weights_only=False)["detector"])
    bank=ConcealBank(list(cfg.concealer.operators),cfg.concealer.strength_levels)
    rows,dd=suppression_curve(det,build_dataset(cfg,"test"),bank,cfg,device,a.points,a.operator)
    for s,au,ee in rows: print(f"s={s:.2f} AUC={au:.4f} EER={ee:.4f}")
    print(f"Δ-drop={dd:.4f}")
    np.savetxt("outputs/curve.csv",np.array([[s,au,ee] for s,au,ee in rows]),
               delimiter=",",header="strength,auc,eer",comments="") if os.makedirs("outputs",exist_ok=True) or True else None
    logger.finish()
if __name__=="__main__": main()
