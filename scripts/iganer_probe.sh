#!/usr/bin/env bash
# Step 1 — pretrain + freeze the artifact probe (run ONCE before the game).
# Uses the standalone probe trainer; for FF++ images set data.root in the
# standalone config, or run inside the CIFT repo with the wrapper below.
set -e
GPU=${1:-0}
CFG=${2:-configs/iganer/iganer_ffpp.yaml}
CUDA_VISIBLE_DEVICES=$GPU python scripts/train_probe.py \
    --config configs/default.yaml \
    --experiment configs/experiment/iganer_ffpp.yaml \
    data.root="${FFPP_ROOT:-/path/to/datasets/ffpp}" \
    probe.ckpt=outputs/ckpt/artifact_probe.pt
echo "[probe] saved -> outputs/ckpt/artifact_probe.pt"
