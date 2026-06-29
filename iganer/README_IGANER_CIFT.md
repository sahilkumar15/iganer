# IGANER × CIFT — Integration Guide

IGANER (Identity-Gap Adversarial Nash Equilibrium via Reinforcement learning) is
a **training-only concealment game** layered on top of the real CIFT model
(`sahilkumar15/ImageDifussionFake`, the DiffusionFake/ControlNet Lightning code).
Inference stays plain CIFT — no Concealer, no probe, no donor.

There are **two modes**. Develop in standalone mode, run real numbers in CIFT mode.

---

## Mode 1 — Standalone (verified, runs anywhere)

A stub detector behind the same `BaseDetector` interface lets you smoke-test the
entire RL game, curriculum, reward, and the Table A ablation runner without the
heavy CIFT/SD environment.

```bash
pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q tests/test_smoke.py          # passes
# full 8-cell Table A on synthetic data:
python scripts/run_ablation_tableA.py --config configs/default.yaml \
    --experiment configs/experiment/iganer_ffpp.yaml data.root=null
```
Numbers here are meaningless (stub). This mode is for developing the RL.

---

## Mode 2 — Real CIFT (drop-in)

### 1. Place files in the CIFT repo
Copy into the root of your CIFT clone (next to `train.py`, `cldm/`, `datasets/`):

```
ImageDifussionFake/
├── train.py                      # (existing CIFT)
├── cldm/  datasets/  models/  utils/  share.py   # (existing CIFT)
├── iganer/                       # ← copy this whole package
├── train_iganer.py               # ← copy (IGANER entrypoint, CIFT conventions)
├── configs/iganer/iganer_ffpp.yaml   # ← copy
└── scripts/iganer_*.sh           # ← copy
```

`iganer/detector/cift_adapter.py` is the only file that knows CIFT internals. It
maps the IGANER interface to CIFT:
| IGANER method | CIFT mapping |
|---|---|
| `logits(x)` | `control_model.fc` on pooled feature-filter features (source-free path) |
| `extract_state(x)` | pooled 1792-d EfficientNet/ConvNeXt features (policy state) |
| `identity_gap(x)` | `DualIdentityMambaFusion` target embedding norm (donor-free Δ proxy) |
| `compute_loss(x,y)` | CIFT `_focal_bce` (0.6 BCE / 0.4 focal) + gap term |

> **One confirmation point.** The Δ readout uses `control_model.mamba_head.dual_mamba`
> (`proj_t`, `norm_t`). If your `mamba_head` exposes a dedicated `gap_readout(feat)`
> or returns `delta`, call that in `cift_adapter.identity_gap()` instead of the
> norm proxy — it's marked `=== CONFIRM ===` in the file.

### 2. Point the config at your data + checkpoint
Edit `configs/iganer/iganer_ffpp.yaml`:
- `dataset.ffpp_rela.data_root` → your FF++ root (same layout CIFT uses)
- `detector.cift_ckpt` → a trained CIFT checkpoint (optional; else SD1.5 init at
  `./models/control_sd15_ini.ckpt` as in `train.py`)
- `model.backbone` → matches your CIFT run (`convnextv2_base` default)

### 3. Run (CIFT conventions: `-c config`, `CUDA_VISIBLE_DEVICES`)
```bash
# Step 1 — pretrain + freeze the artifact probe (once)
FFPP_ROOT=/path/to/ffpp bash scripts/iganer_probe.sh 0

# Step 2 — train ONE variant (full IGANER)
bash scripts/iganer_train.sh 0 configs/iganer/iganer_ffpp.yaml iganer

# Step 3 — the full Table A factorial (all 8 cells) -> outputs/table_A.csv
bash scripts/iganer_tableA.sh 0 configs/iganer/iganer_ffpp.yaml
#   quick decisive check (baseline vs IGANER only):
bash scripts/iganer_tableA.sh 0 configs/iganer/iganer_ffpp.yaml --only 1,8

# Step 4 — suppression-invariance curve (equilibrium certificate)
bash scripts/iganer_eval_curve.sh 0 outputs/ckpt/v8_iganer.pt fft_lowpass
```

### What Table A reports
Per cell: **AvgAUC** and **Δ-drop** (AUC at suppression s=0 minus s=1; lower =
more rendering-invariant). The decisive comparison is **row 2 (game only) vs row
8 (IGANER)** — if row 8's Δ-drop is near-flat while row 2 is steep, the
targeting+protection interaction is load-bearing (the contribution beyond
CRDA-style RL).

---

## Requirements (Mode 2)
The CIFT environment: PyTorch + pytorch-lightning, timm, `mamba_ssm`, the SD1.5 +
ControlNet init checkpoint, and a ≥40 GB GPU (same as CIFT training). The RL game
is training-only; deployed inference is unchanged CIFT.

## Notes
- `game.use_ema: false` in the config avoids deep-copying the full SD model as the
  reward target (recommended for CIFT mode). Set `true` for the lighter stub.
- For multi-GPU, the RL game currently runs single-process; scale the detector
  batch via `train.batch_size`. (CIFT's own `train.py` keeps its torchrun path.)
