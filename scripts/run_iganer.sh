#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_iganer.sh — Full IGANER training script
#
# USAGE:
#   # Full Table A (all 8 cells):
#   bash run_iganer.sh
#
#   # Quick decisive check (baseline vs full IGANER only):
#   bash run_iganer.sh --only 1,8
#
#   # Single variant (full IGANER):
#   bash run_iganer.sh --mode single --tag iganer
#
#   # Smoke test (20 steps per variant, no wandb):
#   bash run_iganer.sh --smoke
#
# OUTPUTS:
#   Checkpoints : $IGANER_ROOT/outputs/ckpt/<variant>.pt
#   Table A CSV : $IGANER_ROOT/outputs/table_A.csv
#   W&B logs    : https://wandb.ai/<your_entity>/IGANER_ICLR
#   Probe ckpt  : $IGANER_ROOT/outputs/ckpt/artifact_probe.pt
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Paths (edit these if your layout changes) ──────────────────────────────────
IGANER_ROOT="/scratch/sahil/projects/img_deepfake/code/iganer"
IGANER_PKG="${IGANER_ROOT}/iganer"                   # contains the iganer/ Python package
CIFT_ROOT="/scratch/sahil/projects/img_deepfake/code/ImageDifussionFake"
CONFIG="${IGANER_PKG}/configs/train_general.yaml"
PROBE_CKPT="${IGANER_ROOT}/outputs/ckpt/artifact_probe.pt"

# ── GPU (comma-separated, e.g. "0,1,2,3" or "4,5,6,7") ───────────────────────
GPUS="4,5,6,7"

# ── Training defaults (override via CLI or edit here) ─────────────────────────
EPOCHS=80
BATCH=32
NUM_WORKERS=4
LR="2e-5"
USE_LPIPS="true"
WANDB="true"

# ── Defaults for mode/tag/only ─────────────────────────────────────────────────
MODE="tableA"
TAG="iganer"
ONLY=""
SMOKE=0
EXTRA_OVERRIDES=""

# ── Parse CLI args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --only)      ONLY="$2";   shift 2 ;;
        --mode)      MODE="$2";   shift 2 ;;
        --tag)       TAG="$2";    shift 2 ;;
        --gpus)      GPUS="$2";   shift 2 ;;
        --epochs)    EPOCHS="$2"; shift 2 ;;
        --smoke)     SMOKE=1;     shift   ;;
        --no-wandb)  WANDB="false"; shift ;;
        *)           EXTRA_OVERRIDES="$EXTRA_OVERRIDES $1"; shift ;;
    esac
done

# ── Smoke test overrides ───────────────────────────────────────────────────────
if [[ $SMOKE -eq 1 ]]; then
    echo "[iganer] SMOKE TEST MODE — 20 steps per variant, wandb off"
    EPOCHS=1
    NUM_WORKERS=0
    USE_LPIPS="false"
    WANDB="false"
    EXTRA_OVERRIDES="$EXTRA_OVERRIDES train.max_steps=20"
fi

# ── Environment ───────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="${IGANER_PKG}:${CIFT_ROOT}:${PYTHONPATH:-}"

# ── Print summary ─────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════"
echo " IGANER Training"
echo "═══════════════════════════════════════════════════════"
echo " Root       : $IGANER_ROOT"
echo " Config     : $CONFIG"
echo " GPUs       : $GPUS"
echo " Mode       : $MODE"
echo " Tag        : $TAG"
echo " Only       : ${ONLY:-all cells}"
echo " Epochs     : $EPOCHS"
echo " Batch      : $BATCH"
echo " Workers    : $NUM_WORKERS"
echo " LPIPS      : $USE_LPIPS"
echo " W&B        : $WANDB"
echo " Outputs    : $IGANER_ROOT/outputs/"
echo " Extra      : $EXTRA_OVERRIDES"
echo "═══════════════════════════════════════════════════════"

# ── Create output dirs ────────────────────────────────────────────────────────
mkdir -p "${IGANER_ROOT}/outputs/ckpt"
mkdir -p "${IGANER_ROOT}/outputs/logs"

# ── Step 1: Pretrain the artifact probe (skip if already done) ────────────────
if [[ ! -f "$PROBE_CKPT" ]]; then
    echo ""
    echo "─── Step 1: Pretraining artifact probe ───────────────────"
    python "${IGANER_PKG}/scripts/train_probe.py" \
        --config "${IGANER_PKG}/configs/default.yaml" \
        --experiment "${IGANER_PKG}/configs/experiment/iganer_ffpp.yaml" \
        probe.ckpt="$PROBE_CKPT" \
        wandb.enabled=false \
        data.num_workers="$NUM_WORKERS" \
        device=cuda
    echo "[probe] saved -> $PROBE_CKPT"
else
    echo ""
    echo "─── Step 1: Probe already exists, skipping ───────────────"
    echo " $PROBE_CKPT"
fi

# ── Step 2: Main IGANER training ──────────────────────────────────────────────
echo ""
echo "─── Step 2: IGANER concealment game training ─────────────"

cd "$IGANER_ROOT"

# Build the python command
CMD="python train_iganer.py"
CMD="$CMD -c $CONFIG"
CMD="$CMD --mode $MODE"
CMD="$CMD --tag $TAG"
[[ -n "$ONLY" ]] && CMD="$CMD --only $ONLY"
CMD="$CMD train.epochs=$EPOCHS"
CMD="$CMD train.batch_size=$BATCH"
CMD="$CMD train.num_workers=$NUM_WORKERS"
CMD="$CMD data.num_workers=$NUM_WORKERS"
CMD="$CMD train.lr=$LR"
CMD="$CMD reward.use_lpips=$USE_LPIPS"
CMD="$CMD wandb.enabled=$WANDB"
CMD="$CMD device=cuda"
CMD="$CMD probe.ckpt=$PROBE_CKPT"
[[ -n "$EXTRA_OVERRIDES" ]] && CMD="$CMD $EXTRA_OVERRIDES"

echo " Running: $CMD"
echo ""
eval $CMD

# ── Step 3: Suppression curve for the best variant ───────────────────────────
BEST_CKPT="${IGANER_ROOT}/outputs/ckpt/v8_iganer.pt"
if [[ "$MODE" == "tableA" && -f "$BEST_CKPT" ]]; then
    echo ""
    echo "─── Step 3: Suppression-invariance curve (v8_iganer) ─────"
    python "${IGANER_PKG}/scripts/eval_suppression_curve.py" \
        --config "${IGANER_PKG}/configs/default.yaml" \
        --ckpt "$BEST_CKPT" \
        --operator fft_lowpass \
        --points 11 \
        device=cuda \
        data.num_workers="$NUM_WORKERS"
    echo "[eval] curve saved -> ${IGANER_ROOT}/outputs/curve.csv"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo " DONE"
echo " Checkpoints : ${IGANER_ROOT}/outputs/ckpt/"
echo " Table A CSV : ${IGANER_ROOT}/outputs/table_A.csv"
echo " Curve CSV   : ${IGANER_ROOT}/outputs/curve.csv"
echo "═══════════════════════════════════════════════════════"



# =========================================================================================================
# chmod +x /scratch/sahil/projects/img_deepfake/code/iganer/scripts/run_iganer.sh
# bash scripts/run_iganer.sh --only 8