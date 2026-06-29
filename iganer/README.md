# IGANER

**Identity-Gap Adversarial Nash Equilibrium via Reinforcement learning**
for rendering-invariant deepfake detection.

A Concealer (PPO policy) plays a two-player game against a Detector (CIFT): the
Concealer hides whatever evidence the Detector attends to; the Detector must
adapt. At the Nash equilibrium, artifact shortcuts are erased and only the
rendering-invariant identity gap carries detection. Inference is the Detector
alone — source-free, no Concealer, no probe.

This repo runs the **Table A factorial ablation** out of the box: all 8 cells of
the three factors that define the method.

## The three factors (Table A)
| | Factor | ON | OFF |
|---|---|---|---|
| **A** | Nash game | learned PPO concealer | fixed random suppression |
| **B** | Targeting | suppression masked to detector saliency | uniform suppression |
| **C** | Protection | identity-gap preserved (reward + detector) | no protection |

All 8 combinations (2³) are produced by `scripts/run_ablation_tableA.py`.

## Project structure
```
iganer/
├── configs/
│   ├── default.yaml                 # all hyperparameters + the 3 factor toggles
│   └── experiment/iganer_ffpp.yaml
├── iganer/
│   ├── data/ffpp.py                 # FF++ loader (+ synthetic fallback)
│   ├── detector/interface.py        # BaseDetector contract + build_detector
│   ├── detector/stub.py             # runnable stand-in (replace with CIFT)
│   ├── concealer/operators.py       # suppression action space (maskable)
│   ├── concealer/attribution.py     # saliency = where the detector looks (B)
│   ├── concealer/policy.py          # PPO concealer (A)
│   ├── probes/artifact_probe.py     # frozen verification probe
│   ├── game/reward.py               # concealer reward; protection gated by C
│   ├── game/ppo.py  ema.py  replay.py  curriculum.py
│   ├── trainer.py                   # ★ factor-aware train_variant() — the core
│   ├── eval.py                      # suppression curve + Δ-drop
│   └── utils/                       # config, seed, metrics, W&B
├── scripts/
│   ├── train_probe.py               # 1) pretrain + freeze probe
│   ├── train.py                     # train ONE variant (set factors on CLI)
│   ├── run_ablation_tableA.py       # ★ run ALL 8 cells -> Table A
│   └── eval_suppression_curve.py
└── tests/test_smoke.py              # end-to-end on synthetic data
```

## Install
```bash
conda create -n iganer python=3.10 -y && conda activate iganer
pip install -r requirements.txt
```

## Data (FF++)
EULA: <https://github.com/ondyari/FaceForensics>. Pre-extracted frames:
<https://github.com/SCLBD/DeepfakeBench>. Point `data.root` at the frames root
(layout in `iganer/data/ffpp.py`). Leave `data.root=null` for smoke testing.

## Run — the Table A ablation
```bash
# 0) smoke test (no data, no GPU)
PYTHONPATH=. python -m pytest -q tests/test_smoke.py

# 1) pretrain the frozen artifact probe
python scripts/train_probe.py --config configs/default.yaml \
    --experiment configs/experiment/iganer_ffpp.yaml data.root=/data/FFpp

# 2) run ALL 8 ablation cells -> prints Table A, saves outputs/table_A.csv
python scripts/run_ablation_tableA.py --config configs/default.yaml \
    --experiment configs/experiment/iganer_ffpp.yaml \
    data.root=/data/FFpp train.epochs=80 wandb.enabled=true

# quick subset (baseline vs IGANER only):
python scripts/run_ablation_tableA.py --config configs/default.yaml \
    --experiment configs/experiment/iganer_ffpp.yaml --only 1,8 data.root=/data/FFpp

# train a single variant by hand:
python scripts/train.py --config configs/default.yaml \
    factors.game=true factors.targeting=true factors.protection=true \
    --tag iganer data.root=/data/FFpp
```

## What Table A reports
For each of the 8 cells: **Avg AUC** (detection performance) and **Δ-drop**
(AUC at suppression s=0 minus s=1 — lower means more rendering-invariant). The
decisive comparison is **row 2 (game only) vs row 8 (IGANER)**: if row 8 has a
near-flat Δ-drop while row 2 is steep, the targeting+protection combination is
load-bearing — the contribution beyond CRDA-style RL.

## Wiring in real CIFT
Implement `CIFTAdapter(BaseDetector)` (`logits`, `extract_state`, `identity_gap`
= source-free Δ, `compute_loss` = focal CE + XID/IGS) at the marked point in
`iganer/detector/interface.py`, then set `detector.type=cift`. Nothing in the
game / trainer / ablation code changes.

## Status
The `stub` detector makes everything run today; its numbers are **meaningless**.
The Δ-drop column only becomes meaningful with real CIFT + FF++ and full epochs.
