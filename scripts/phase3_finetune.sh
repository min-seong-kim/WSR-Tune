#!/usr/bin/env bash
# Phase 3: choose the downstream task and training settings below.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Edit the defaults after :- here, or override them with environment variables.
# Shared model, basis/mask paths, precision, sequence length, and seed are in common.sh.
TASK="${TASK:-gsm8k}"                 # gsm8k | math | arc_challenge | medqa
TASK_PATH="${TASK_PATH:-}"            # Optional local JSON/JSONL; empty downloads the dataset.
TASK_SAMPLES="${TASK_SAMPLES:-}"      # Empty: task default below; 0: full split; N: sample N rows.
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}" # Use 1e-5 for the paper's Gemma-2-9B-IT runs.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"         # Effective batch size = TRAIN_BATCH_SIZE * GRAD_ACCUM on one GPU.
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}" # true | false
LOGGING_STEPS="${LOGGING_STEPS:-10}"
MAX_STEPS="${MAX_STEPS:--1}"           # -1: use EPOCHS; positive values override EPOCHS.

# Use the full GSM8K, MATH, and ARC-Challenge training sets by default,
# and randomly sample 10,000 MedQA training questions using the shared SEED.
case "$TASK" in
    gsm8k|math|arc_challenge) TASK_SAMPLES="${TASK_SAMPLES:-0}" ;;
    medqa) TASK_SAMPLES="${TASK_SAMPLES:-10000}" ;;
    *) echo "Unknown TASK=$TASK (use gsm8k, math, arc_challenge, or medqa)" >&2; exit 1 ;;
esac

TASK_ARGS=(--task "$TASK" --task_samples "$TASK_SAMPLES")
if [[ -n "$TASK_PATH" ]]; then
    TASK_ARGS+=(--task_path "$TASK_PATH")
fi

TRAIN_ARGS=()
case "$GRADIENT_CHECKPOINTING" in
    true) TRAIN_ARGS+=(--gradient_checkpointing) ;;
    false) ;;
    *) echo "Unknown GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING (use true or false)" >&2; exit 1 ;;
esac

"$PYTHON" train.py --phase 3 "${COMMON_ARGS[@]}" \
    --basis_dir "$BASIS_DIR" --masks_dir "$MASKS_DIR" \
    "${TASK_ARGS[@]}" \
    --epochs "$EPOCHS" --learning_rate "$LEARNING_RATE" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --weight_decay "$WEIGHT_DECAY" --warmup_ratio "$WARMUP_RATIO" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --logging_steps "$LOGGING_STEPS" --max_steps "$MAX_STEPS" \
    "${TRAIN_ARGS[@]}" --output_dir "$RUN_DIR/phase3_$TASK" "$@"
