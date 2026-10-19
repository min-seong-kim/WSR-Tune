#!/usr/bin/env bash
# Shared settings keep the model and target projections consistent across phases.
set -euo pipefail

WSR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WSR_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${PYTHON:-python}"
MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
RUN_DIR="${RUN_DIR:-outputs/wsr_tune}"
SAFETY_DATA="${SAFETY_DATA:-data/safety_calibration.json}"
SAFETY_SAMPLES="${SAFETY_SAMPLES:-4994}"
LAYER_TYPES="${LAYER_TYPES:-attn_q,attn_k,attn_v,ffn_up,ffn_down}"
TARGET_LAYERS="${TARGET_LAYERS:-all}"
BASIS_DIR="${BASIS_DIR:-$RUN_DIR/phase1/basis}"
MASKS_DIR="${MASKS_DIR:-$RUN_DIR/phase2/masks}"
COMMON_ARGS=(
    --model "$MODEL"
    --layer_types "$LAYER_TYPES"
    --target_layers "$TARGET_LAYERS"
    --device "${DEVICE:-cuda}"
    --dtype "${DTYPE:-bfloat16}"
    --chat_format "${CHAT_FORMAT:-auto}"
    --max_length "${MAX_LENGTH:-1024}"
    --seed "${SEED:-42}"
)
if [[ -n "${REVISION:-}" ]]; then
    COMMON_ARGS+=(--revision "$REVISION")
fi
