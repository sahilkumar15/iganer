#!/usr/bin/env python
"""Train a SINGLE ablation variant. Set factors on the CLI.

  # full IGANER:
  python scripts/train.py --config configs/default.yaml \
      --experiment configs/experiment/iganer_ffpp.yaml \
      factors.game=true factors.targeting=true factors.protection=true data.root=/data/FFpp
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from iganer.utils.config import load_config
from iganer.utils.seed import set_seed
from iganer.utils.logging import Logger
from iganer.trainer import train_variant
from iganer.eval import suppression_curve
from iganer.data.ffpp import build_dataset

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/default.yaml")
    ap.add_argument("--experiment",default=None); ap.add_argument("--tag",default="single")
    ap.add_argument("overrides",nargs="*"); a=ap.parse_args()
    cfg=load_config(a.config,a.experiment,a.overrides); set_seed(cfg.seed)
    logger=Logger(cfg)
    detector,bank,m=train_variant(cfg,logger,a.tag)
    device=cfg.device if __import__("torch").cuda.is_available() else "cpu"
    _,dd=suppression_curve(detector,build_dataset(cfg,"test"),bank,cfg,device,cfg.eval.curve_points)
    print(f"[{a.tag}] val_auc={m['val_auc']:.4f} val_eer={m['val_eer']:.4f} delta_drop={dd:.4f}")
    logger.log({"val_auc":m['val_auc'],"delta_drop":dd}); logger.finish()
if __name__=="__main__": main()
