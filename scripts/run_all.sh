#!/usr/bin/env bash
# Phase 1 -> Phase 2 -> Phase 3 with the task/settings in phase3_finetune.sh.
# The basis and masks are task-independent; rerun only phase3_finetune.sh for another task.
set -euo pipefail
WSR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$WSR_SCRIPT_DIR/phase1_basis.sh"
bash "$WSR_SCRIPT_DIR/phase2_mask.sh"
bash "$WSR_SCRIPT_DIR/phase3_finetune.sh"
