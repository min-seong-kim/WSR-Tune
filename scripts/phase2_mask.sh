#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

"$PYTHON" train.py --phase 2 "${COMMON_ARGS[@]}" \
    --basis_dir "$BASIS_DIR" \
    --safety_data "$SAFETY_DATA" --safety_samples "$SAFETY_SAMPLES" \
    --batch_size "${CALIBRATION_BATCH_SIZE:-2}" \
    --keep_ratio "${KEEP_RATIO:-0.1}" \
    --output_dir "$RUN_DIR/phase2" "$@"
