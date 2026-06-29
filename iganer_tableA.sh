#!/usr/bin/env bash
# Step 3 — run the FULL Table A factorial (all 8 cells) on the real CIFT model.
# Each cell trains a fresh CIFT detector under one (A,B,C) setting, then computes
# AvgAUC + Δ-drop. Prints Table A and writes outputs/table_A.csv.
#   bash scripts/iganer_tableA.sh 0 configs/iganer/iganer_ffpp.yaml
#   quick check (baseline vs IGANER): ... --only 1,8   (append after config)
set -e
GPU=${1:-0}; CFG=${2:-configs/iganer/iganer_ffpp.yaml}; shift 2 || true
CUDA_VISIBLE_DEVICES=$GPU python train_iganer.py -c "$CFG" --mode tableA "$@"
