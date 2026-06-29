#!/usr/bin/env bash
# Step 4 — suppression-invariance curve (the equilibrium certificate) for a ckpt.
#   bash scripts/iganer_eval_curve.sh 0 outputs/ckpt/v8_iganer.pt fft_lowpass
set -e
GPU=${1:-0}; CKPT=${2:-outputs/ckpt/v8_iganer.pt}; OP=${3:-fft_lowpass}
CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_suppression_curve.py \
    --config configs/default.yaml --ckpt "$CKPT" --operator "$OP" --points 11
