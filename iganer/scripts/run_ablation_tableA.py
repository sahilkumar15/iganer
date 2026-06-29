#!/usr/bin/env python
"""Run the full Table A factorial ablation: all 8 cells of (game, targeting,
protection). Trains each, computes Avg AUC and Δ-drop, prints + saves Table A.

  python scripts/run_ablation_tableA.py --config configs/default.yaml \
      --experiment configs/experiment/iganer_ffpp.yaml data.root=/data/FFpp

  # subset (e.g. baseline vs IGANER) for a quick check:
  python scripts/run_ablation_tableA.py ... --only 1,8
"""
import argparse, os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from omegaconf import OmegaConf
from iganer.utils.config import load_config
from iganer.utils.seed import set_seed
from iganer.utils.logging import Logger
from iganer.probes.artifact_probe import ArtifactProbe
from iganer.trainer import train_variant
from iganer.eval import suppression_curve
from iganer.data.ffpp import build_dataset

# (id, tag, game A, targeting B, protection C)
VARIANTS = [
    (1, "v1_baseline",   False, False, False),
    (2, "v2_game",       True,  False, False),
    (3, "v3_target",     False, True,  False),
    (4, "v4_protect",    False, False, True ),
    (5, "v5_game_tgt",   True,  True,  False),
    (6, "v6_game_prot",  True,  False, True ),
    (7, "v7_tgt_prot",   False, True,  True ),
    (8, "v8_iganer",     True,  True,  True ),
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/default.yaml")
    ap.add_argument("--experiment",default=None)
    ap.add_argument("--only",default=None,help="comma ids, e.g. 1,8")
    ap.add_argument("overrides",nargs="*"); a=ap.parse_args()
    cfg=load_config(a.config,a.experiment,a.overrides); set_seed(cfg.seed)
    logger=Logger(cfg)
    device=cfg.device if torch.cuda.is_available() else "cpu"

    # shared frozen probe (pretrained once) so all cells compare fairly
    probe=ArtifactProbe().to(device)
    if cfg.probe.ckpt and os.path.exists(cfg.probe.ckpt):
        probe.load_state_dict(torch.load(cfg.probe.ckpt,map_location=device))
    else:
        print("[probe] random init; run train_probe.py for real runs.")
    probe.freeze()

    keep=set(int(i) for i in a.only.split(",")) if a.only else None
    rows=[]
    for vid,tag,A,B,C in VARIANTS:
        if keep and vid not in keep: continue
        set_seed(cfg.seed)                       # identical init across cells
        cfg.factors=OmegaConf.create(dict(game=A,targeting=B,protection=C))
        print(f"\n=== variant {vid}: {tag}  (A={A} B={B} C={C}) ===")
        detector,bank,m=train_variant(cfg,logger,tag,probe=probe)
        _,dd=suppression_curve(detector,build_dataset(cfg,"test"),bank,cfg,device,cfg.eval.curve_points)
        tick=lambda v:"✓" if v else "✗"
        rows.append([vid,tag,tick(A),tick(B),tick(C),round(m["val_auc"],4),round(dd,4)])
        logger.log({"variant":vid,"avg_auc":m["val_auc"],"delta_drop":dd})

    print("\n================ TABLE A ================")
    hdr=["#","variant","A:game","B:target","C:protect","AvgAUC","Δ-drop"]
    print("  ".join(f"{h:>10}" for h in hdr))
    for r in rows: print("  ".join(f"{str(c):>10}" for c in r))
    os.makedirs("outputs",exist_ok=True)
    with open("outputs/table_A.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(hdr); w.writerows(rows)
    logger.log_table("table_A",hdr,rows); logger.finish()
    print("\nsaved -> outputs/table_A.csv")

if __name__=="__main__": main()
