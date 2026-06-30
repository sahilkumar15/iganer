#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# run_iganer.sh — IGANER full training script (generalized)
#
# USAGE (always run from iganer root):
#   bash scripts/run_iganer.sh                        # full Table A, all 8 cells
#   bash scripts/run_iganer.sh --only 1,8             # baseline vs full IGANER
#   bash scripts/run_iganer.sh --only 8               # full IGANER only
#   bash scripts/run_iganer.sh --smoke                # 20-step sanity check
#   bash scripts/run_iganer.sh --smoke --only 1,8     # fast sanity check
#   bash scripts/run_iganer.sh --gpus 0,1,2,3         # choose GPUs
#   bash scripts/run_iganer.sh --gpus 4               # single GPU
#   bash scripts/run_iganer.sh --batch 8              # smaller batch (OOM fix)
#   bash scripts/run_iganer.sh --epochs 20            # fewer epochs
#   bash scripts/run_iganer.sh --no-wandb             # disable W&B logging
#   bash scripts/run_iganer.sh --mode single --tag v8 # single variant
#
# OUTPUTS:
#   outputs/ckpt/<variant>.pt   — model checkpoints
#   outputs/table_A.csv         — Table A results
#   outputs/curve.csv           — suppression-invariance curve
#   outputs/ckpt/artifact_probe.pt
#   W&B: https://wandb.ai/sahilthegnius/IGANER_ICLR
#
# COMMON RECIPES:
#   # OOM fix (80GB A100, single GPU):
#   bash scripts/run_iganer.sh --gpus 1,2,3,4 --only 2,3,4,5,6,7
#
#   # Full Table A on 4 GPUs (one cell per GPU sequentially):
#   bash scripts/run_iganer.sh --gpus 4,5,6,7 --batch 8
#
#   # Quick result: baseline vs IGANER, no W&B:
#   bash scripts/run_iganer.sh --gpus 4 --batch 8 --only 1,8 --no-wandb
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — edit CIFT_ROOT if your CIFT repo is elsewhere; everything else is
# auto-resolved relative to this script's location.
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IGANER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IGANER_PKG="${IGANER_ROOT}/iganer"
IGANER_SCRIPTS="${IGANER_ROOT}/scripts"
IGANER_CONFIGS="${IGANER_PKG}/configs"
CIFT_ROOT="/scratch/sahil/projects/img_deepfake/code/ImageDifussionFake"

TRAIN_ENTRYPOINT="${IGANER_ROOT}/train_iganer.py"
CONFIG="${IGANER_CONFIGS}/train_general.yaml"
PROBE_CKPT="${IGANER_ROOT}/outputs/ckpt/artifact_probe.pt"
BEST_CKPT="${IGANER_ROOT}/outputs/ckpt/v8_iganer.pt"

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS — all overridable via CLI flags
# ─────────────────────────────────────────────────────────────────────────────
GPUS="4,5,6,7"       # CUDA_VISIBLE_DEVICES  (e.g. "0" / "0,1" / "4,5,6,7")
EPOCHS=20             # training epochs per variant
BATCH=32              # per-GPU batch size (8 safe for 80GB A100 + CIFT 860M)
NUM_WORKERS=4         # dataloader workers (set 0 if hangs)
LR="2e-5"             # detector learning rate
USE_LPIPS="true"      # LPIPS realism reward (set false to save ~2GB)
WANDB="true"          # W&B logging
MODE="tableA"         # tableA | single
TAG="iganer"          # checkpoint tag for --mode single
ONLY=""               # e.g. "1,8" — subset of 8 variants; empty = all
SMOKE=0               # 1 = 20-step smoke test
EXTRA_OVERRIDES=""    # any extra omegaconf overrides

# ─────────────────────────────────────────────────────────────────────────────
# PARSE CLI
# ─────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)      GPUS="$2";         shift 2 ;;
        --epochs)    EPOCHS="$2";       shift 2 ;;
        --batch)     BATCH="$2";        shift 2 ;;
        --workers)   NUM_WORKERS="$2";  shift 2 ;;
        --lr)        LR="$2";           shift 2 ;;
        --only)      ONLY="$2";         shift 2 ;;
        --mode)      MODE="$2";         shift 2 ;;
        --tag)       TAG="$2";          shift 2 ;;
        --no-wandb)  WANDB="false";     shift   ;;
        --no-lpips)  USE_LPIPS="false"; shift   ;;
        --smoke)     SMOKE=1;           shift   ;;
        --cift-root) CIFT_ROOT="$2";    shift 2 ;;
        --config)    CONFIG="$2";       shift 2 ;;
        *)           EXTRA_OVERRIDES="$EXTRA_OVERRIDES $1"; shift ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST OVERRIDES
# ─────────────────────────────────────────────────────────────────────────────
if [[ $SMOKE -eq 1 ]]; then
    EPOCHS=1
    NUM_WORKERS=0
    USE_LPIPS="false"
    WANDB="false"
    EXTRA_OVERRIDES="$EXTRA_OVERRIDES train.max_steps=20"
    echo "[iganer] SMOKE TEST — 20 steps/variant, wandb off, lpips off"
fi

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATE REQUIRED PATHS
# ─────────────────────────────────────────────────────────────────────────────
MISSING=0
for p in \
    "$IGANER_PKG" \
    "$CIFT_ROOT" \
    "$CONFIG" \
    "$TRAIN_ENTRYPOINT" \
    "$IGANER_SCRIPTS/train_probe.py" \
    "$IGANER_SCRIPTS/eval_suppression_curve.py"; do
    if [[ ! -e "$p" ]]; then
        echo "[ERROR] Missing path: $p"
        MISSING=1
    fi
done
if [[ $MISSING -eq 1 ]]; then
    echo ""
    echo "Fix missing paths above, then re-run."
    echo "If CIFT is elsewhere: bash scripts/run_iganer.sh --cift-root /your/path"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="${IGANER_PKG}:${CIFT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " IGANER Training"
echo "═══════════════════════════════════════════════════════════"
echo " Root        : $IGANER_ROOT"
echo " Config      : $CONFIG"
echo " CIFT root   : $CIFT_ROOT"
echo " GPUs        : $GPUS  (CUDA_VISIBLE_DEVICES)"
echo " Mode        : $MODE"
echo " Tag         : $TAG"
echo " Only        : ${ONLY:-all 8 cells}"
echo " Epochs      : $EPOCHS"
echo " Batch       : $BATCH"
echo " Workers     : $NUM_WORKERS"
echo " LR          : $LR"
echo " LPIPS       : $USE_LPIPS"
echo " W&B         : $WANDB"
echo " Smoke       : $([[ $SMOKE -eq 1 ]] && echo YES || echo no)"
echo " Outputs     : $IGANER_ROOT/outputs/"
[[ -n "${EXTRA_OVERRIDES// /}" ]] && echo " Extra       :$EXTRA_OVERRIDES"
echo "═══════════════════════════════════════════════════════════"

mkdir -p "${IGANER_ROOT}/outputs/ckpt"
mkdir -p "${IGANER_ROOT}/outputs/logs"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — artifact probe (skip if checkpoint already exists)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
if [[ -f "$PROBE_CKPT" ]]; then
    echo "─── Step 1: Probe exists, skipping ──────────────────────────"
    echo "    $PROBE_CKPT"
else
    echo "─── Step 1: Pretraining artifact probe ──────────────────────"
    python "${IGANER_SCRIPTS}/train_probe.py" \
        --config "${CONFIG}" \
        probe.ckpt="${PROBE_CKPT}" \
        wandb.enabled=false \
        data.num_workers="${NUM_WORKERS}" \
        device=cuda \
        data.root=null
    echo "[probe] saved -> $PROBE_CKPT"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — IGANER concealment game
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "─── Step 2: IGANER concealment game ─────────────────────────"
cd "$IGANER_ROOT"

CMD=(
    python "${TRAIN_ENTRYPOINT}"
    -c "${CONFIG}"
    --mode "${MODE}"
    --tag  "${TAG}"
    "train.epochs=${EPOCHS}"
    "train.batch_size=${BATCH}"
    "train.num_workers=${NUM_WORKERS}"
    "data.num_workers=${NUM_WORKERS}"
    "train.lr=${LR}"
    "train.grad_clip_val=0.5"
    "reward.use_lpips=${USE_LPIPS}"
    "wandb.enabled=${WANDB}"
    "device=cuda"
    "probe.ckpt=${PROBE_CKPT}"
)
[[ -n "$ONLY" ]]                  && CMD+=(--only "${ONLY}")
[[ -n "${EXTRA_OVERRIDES// /}" ]] && CMD+=($EXTRA_OVERRIDES)

echo " ${CMD[*]}"
echo ""
"${CMD[@]}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — suppression-invariance curve (only after tableA with v8 present)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
if [[ "$MODE" == "tableA" && -f "$BEST_CKPT" ]]; then
    echo "─── Step 3: Suppression curve (v8_iganer) ───────────────────"
    python "${IGANER_SCRIPTS}/eval_suppression_curve.py" \
        --config "${CONFIG}" \
        --ckpt   "${BEST_CKPT}" \
        --operator fft_lowpass \
        --points 11 \
        device=cuda \
        data.num_workers="${NUM_WORKERS}" \
        data.root=null
    echo "[eval] curve -> ${IGANER_ROOT}/outputs/curve.csv"
elif [[ "$MODE" == "tableA" ]]; then
    echo "─── Step 3: Skipped (v8_iganer.pt not found) ────────────────"
fi

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo " DONE"
echo " Checkpoints:"
ls "${IGANER_ROOT}/outputs/ckpt/" 2>/dev/null | sed 's/^/   /' || true
echo " Table A : ${IGANER_ROOT}/outputs/table_A.csv"
echo " Curve   : ${IGANER_ROOT}/outputs/curve.csv"
echo "═══════════════════════════════════════════════════════════"

# =========================================================================================================
# chmod +x /scratch/sahil/projects/img_deepfake/code/iganer/scripts/run_iganer.sh
# bash scripts/run_iganer.sh --only 8