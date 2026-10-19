#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

"$PYTHON" train.py --phase 1 "${COMMON_ARGS[@]}" \
    --safety_data "$SAFETY_DATA" --safety_samples "$SAFETY_SAMPLES" \
    --batch_size "${CALIBRATION_BATCH_SIZE:-2}" \
    --layers_per_pass "${LAYERS_PER_PASS:-0}" \
    --gram_dtype "${GRAM_DTYPE:-model}" \
    --output_dir "$RUN_DIR/phase1" "$@"
