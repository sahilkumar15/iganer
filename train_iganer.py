#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""IGANER training entrypoint — CIFT conventions.

Lives at the ROOT of the CIFT repo (sahilkumar15/ImageDifussionFake), alongside
train.py, so that `cldm`, `datasets`, `share`, `utils` are importable.

It builds the REAL CIFT model exactly as train.py does, wraps it in CIFTAdapter,
builds CIFT's FF++ dataloaders via the dataset factory, then runs the IGANER
concealment game through iganer.trainer.train_variant.

Run (single variant — full IGANER):
  CUDA_VISIBLE_DEVICES=0 python train_iganer.py -c configs/iganer/iganer_ffpp.yaml \
      --mode single --tag iganer

Run (full Table A factorial — all 8 cells):
  CUDA_VISIBLE_DEVICES=0 python train_iganer.py -c configs/iganer/iganer_ffpp.yaml \
      --mode tableA

NOTE: requires the full CIFT environment (SD1.5 ckpt at ./models/control_sd15_ini.ckpt,
mamba_ssm, timm, pytorch-lightning, a 40G GPU). The RL game itself is training-only;
inference remains plain CIFT.
"""
import os, sys, argparse, csv
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

# IGANER package (place the `iganer/` folder at repo root or on PYTHONPATH)
from iganer.utils.seed import set_seed
from iganer.utils.logging import Logger
from iganer.trainer import train_variant
from iganer.eval import suppression_curve
from iganer.detector.cift_adapter import CIFTAdapter
from iganer.probes.artifact_probe import ArtifactProbe

# CIFT repo modules
import share  # noqa: F401  (CIFT side-effects: disable_verbosity)
from cldm.model import create_model, load_state_dict
from datasets import create_dataset


# ---- 8-cell factorial (game A, targeting B, protection C) ----
VARIANTS = [
    (1, "v1_baseline",  False, False, False),
    (2, "v2_game",      True,  False, False),
    (3, "v3_target",    False, True,  False),
    (4, "v4_protect",   False, False, True ),
    (5, "v5_game_tgt",  True,  True,  False),
    (6, "v6_game_prot", True,  False, True ),
    (7, "v7_tgt_prot",  False, True,  True ),
    (8, "v8_iganer",    True,  True,  True ),
]


class CIFTImageDataset(Dataset):
    """Wrap a CIFT dataset so it yields (image01, label) for IGANER.
    Picks the observed/target face and maps CIFT [-1,1] -> [0,1] BCHW."""
    def __init__(self, base):
        self.base = base
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        s = self.base[i]
        if isinstance(s, (list, tuple)):
            s = s[0]
        img = None
        for k in ("target", "source", "hint", "image", "jpg"):
            if isinstance(s, dict) and k in s:
                img = s[k]; break
        if img is None:
            raise KeyError(f"no image key in CIFT sample: {list(s.keys())}")
        img = torch.as_tensor(img).float()
        if img.ndim == 3 and img.shape[0] != 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1).contiguous()
        mx, mn = float(img.max()), float(img.min())
        if mx > 2.0:            # [0,255]
            img = img / 255.0
        elif mn < 0.0:          # [-1,1]
            img = (img + 1.0) * 0.5
        img = img.clamp(0, 1)
        label = float(s["label"]) if isinstance(s, dict) and "label" in s else 0.0
        return img, label


def build_cift_detector(cfg, device):
    CIFT_ROOT = "/scratch/sahil/projects/img_deepfake/code/ImageDifussionFake"  # adjust if needed
    model_cfg = os.path.join(CIFT_ROOT, "configs/diffusionfake_mixed.yaml")
    if not os.path.isfile(model_cfg):
        model_cfg = os.path.join(CIFT_ROOT, "configs/diffusionfake.yaml")
    model = create_model(model_cfg).cpu()

    model.args = cfg
    backbone = getattr(getattr(cfg, "model", None), "backbone", "convnextv2_base")
    model.control_model.define_feature_filter(backbone)

    # propagate ablation flags (full model if absent) — as in train.py
    abl = getattr(cfg, "ablation", None)
    if abl is not None:
        model.ablation = abl
        model.control_model.ablation = abl

    # load init weights (SD1.5+ControlNet) or a CIFT checkpoint
    init_path = getattr(cfg, "ckpt_path", None) or os.path.join(CIFT_ROOT, "models/control_sd15_ini.ckpt")
    if cfg.detector.get("cift_ckpt", None) and os.path.isfile(cfg.detector.cift_ckpt):
        init_path = cfg.detector.cift_ckpt
    if os.path.isfile(init_path):
        sd = load_state_dict(init_path, location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        sd.pop("cond_stage_model.transformer.text_model.embeddings.position_ids", None)
        ms = model.state_dict()
        filt = {k: v for k, v in sd.items() if k in ms and ms[k].shape == v.shape}
        model.load_state_dict(filt, strict=False)
        print(f"[CIFT] loaded {len(filt)} keys from {init_path}")

    model.sd_locked = True
    model.only_mid_control = False
    return CIFTAdapter(model, cfg).to(device)


def make_loaders(cfg):
    base_train = create_dataset(cfg, split="train").dataset
    base_val   = create_dataset(cfg, split="val").dataset
    train_ds = CIFTImageDataset(base_train)
    val_ds   = CIFTImageDataset(base_val)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.train.batch_size), shuffle=True,
                              num_workers=int(cfg.train.num_workers), drop_last=True,
                              pin_memory=True)
    return train_loader, val_ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--mode", choices=["single", "tableA"], default="single")
    ap.add_argument("--tag", default="iganer")
    ap.add_argument("--only", default=None, help="tableA subset, e.g. 1,8")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()

    cfg = OmegaConf.load(a.config)
    if a.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(a.overrides)))
    set_seed(int(getattr(cfg, "seed", 3407)))
    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    logger = Logger(cfg)

    detector = build_cift_detector(cfg, device)
    train_loader, val_ds = make_loaders(cfg)

    # frozen artifact probe (shared across all cells)
    probe = ArtifactProbe().to(device)
    if cfg.probe.ckpt and os.path.isfile(cfg.probe.ckpt):
        probe.load_state_dict(torch.load(cfg.probe.ckpt, map_location=device))
    else:
        print("[probe] random init; run scripts/iganer_probe.sh first for real runs.")
    probe.freeze()

    if a.mode == "single":
        detector, bank, m = train_variant(cfg, logger, a.tag, probe=probe,
                                          detector=detector,
                                          train_loader=train_loader, val_ds=val_ds)
        _, dd = suppression_curve(detector, val_ds, bank, cfg, device, cfg.eval.curve_points)
        print(f"[{a.tag}] val_auc={m['val_auc']:.4f} delta_drop={dd:.4f}")
        logger.log({"val_auc": m["val_auc"], "delta_drop": dd}); logger.finish()
        return

    # mode == tableA : all 8 cells, fresh CIFT detector per cell
    keep = set(int(i) for i in a.only.split(",")) if a.only else None
    rows = []
    for vid, tag, A, B, C in VARIANTS:
        if keep and vid not in keep:
            continue
        set_seed(int(getattr(cfg, "seed", 3407)))
        cfg.factors = OmegaConf.create(dict(game=A, targeting=B, protection=C))
        det = build_cift_detector(cfg, device)          # fresh weights per cell
        print(f"\n=== variant {vid}: {tag} (A={A} B={B} C={C}) ===")
        det, bank, m = train_variant(cfg, logger, tag, probe=probe,
                                     detector=det, train_loader=train_loader, val_ds=val_ds)
        _, dd = suppression_curve(det, val_ds, bank, cfg, device, cfg.eval.curve_points)
        tick = lambda v: "Y" if v else "N"
        rows.append([vid, tag, tick(A), tick(B), tick(C), round(m["val_auc"], 4), round(dd, 4)])
        logger.log({"variant": vid, "avg_auc": m["val_auc"], "delta_drop": dd})

    print("\n================ TABLE A ================")
    hdr = ["#", "variant", "A:game", "B:target", "C:protect", "AvgAUC", "Delta-drop"]
    print("  ".join(f"{h:>11}" for h in hdr))
    for r in rows:
        print("  ".join(f"{str(c):>11}" for c in r))
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/table_A.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    logger.log_table("table_A", hdr, rows); logger.finish()
    print("\nsaved -> outputs/table_A.csv")


if __name__ == "__main__":
    main()
