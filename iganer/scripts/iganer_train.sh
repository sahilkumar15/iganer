#!/usr/bin/env bash
# Step 2 — train ONE IGANER variant on the real CIFT model.
# Usage: bash scripts/iganer_train.sh <GPU> <CONFIG> <TAG> [factor overrides...]
#   full IGANER:      bash scripts/iganer_train.sh 0 configs/iganer/iganer_ffpp.yaml iganer
#   game-only (v2):   bash scripts/iganer_train.sh 0 configs/iganer/iganer_ffpp.yaml v2_game \
#                        factors.game=true factors.targeting=false factors.protection=false
set -e
GPU=${1:-0}; CFG=${2:-configs/iganer/iganer_ffpp.yaml}; TAG=${3:-iganer}; shift 3 || true
CUDA_VISIBLE_DEVICES=$GPU python train_iganer.py -c "$CFG" --mode single --tag "$TAG" "$@"
