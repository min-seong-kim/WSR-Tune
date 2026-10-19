#!/usr/bin/env python3
"""Run one of the three WSR-Tune phases."""

import argparse
import logging

from wsr_tune.data import TASK_NAMES


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Safety-aligned Hugging Face model ID or local checkpoint.",
    )
    parser.add_argument(
        "--revision", default=None, help="Optional model revision/commit."
    )
    parser.add_argument(
        "--output_dir", required=True, help="An empty directory for this phase."
    )
    parser.add_argument("--layer_types", default="attn_q,attn_k,attn_v,ffn_up,ffn_down")
    parser.add_argument("--target_layers", default="all", help="all, 0-15, or 0,2,4-6.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--chat_format",
        choices=("auto", "chat", "plain"),
        default="auto",
        help="auto uses the tokenizer's chat template when available.",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    calibration = parser.add_argument_group("Phases 1 and 2: safety calibration")
    calibration.add_argument("--safety_data", default="data/safety_calibration.json")
    calibration.add_argument(
        "--safety_samples",
        type=int,
        default=0,
        help="First N calibration examples; 0 uses all.",
    )
    calibration.add_argument(
        "--layers_per_pass",
        type=int,
        default=0,
        help="Phase 1 blocks per data pass; 0 processes all together.",
    )
    calibration.add_argument(
        "--gram_dtype",
        choices=("model", "float32"),
        default="model",
        help="Phase 1 accumulation dtype; SVD always runs in float32.",
    )

    scoring = parser.add_argument_group("Phase 2: importance and masking")
    scoring.add_argument(
        "--basis_dir", help="Phase 1 basis directory (Phases 2 and 3)."
    )
    scoring.add_argument(
        "--keep_ratio",
        type=float,
        default=0.1,
        help="Protected fraction per projection matrix; threshold ties are retained.",
    )

    tuning = parser.add_argument_group("Phase 3: downstream fine-tuning")
    tuning.add_argument("--masks_dir", help="Phase 2 masks directory.")
    tuning.add_argument(
        "--task",
        choices=TASK_NAMES,
        default="gsm8k",
        help="Downstream training set, downloaded from Hugging Face Datasets.",
    )
    tuning.add_argument(
        "--task_path",
        default=None,
        help="Optional local JSON/JSONL copy of the task with the same row schema.",
    )
    tuning.add_argument(
        "--task_samples",
        type=int,
        default=0,
        help="Number of training examples; 0 uses the full train split.",
    )
    tuning.add_argument("--epochs", type=float, default=3)
    tuning.add_argument("--learning_rate", type=float, default=5e-5)
    tuning.add_argument("--gradient_accumulation_steps", type=int, default=8)
    tuning.add_argument("--weight_decay", type=float, default=0.01)
    tuning.add_argument("--warmup_ratio", type=float, default=0.1)
    tuning.add_argument("--max_grad_norm", type=float, default=1.0)
    tuning.add_argument("--gradient_checkpointing", action="store_true")
    tuning.add_argument("--logging_steps", type=int, default=10)
    tuning.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Positive values override epochs, useful for smoke tests.",
    )
    args = parser.parse_args()
    if args.phase in (2, 3) and not args.basis_dir:
        parser.error("--basis_dir is required for Phases 2 and 3.")
    if args.phase == 3 and not args.masks_dir:
        parser.error("--masks_dir is required for Phase 3.")
    for key in ("batch_size", "gradient_accumulation_steps", "logging_steps"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key} must be positive.")
    if args.max_length < 2:
        parser.error("--max_length must be at least 2.")
    if not 0 <= args.keep_ratio <= 1:
        parser.error("--keep_ratio must be between 0 and 1.")
    if not 0 <= args.warmup_ratio <= 1 or args.epochs <= 0 or args.learning_rate <= 0:
        parser.error("Invalid warmup_ratio, epochs, or learning_rate.")
    if args.weight_decay < 0 or args.max_grad_norm < 0:
        parser.error("weight_decay and max_grad_norm must be nonnegative.")
    for key in ("safety_samples", "task_samples", "layers_per_pass"):
        if getattr(args, key) < 0:
            parser.error(f"--{key} must be nonnegative.")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max_steps must be -1 or a positive integer.")
    return args


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.phase == 1:
        from wsr_tune.phase1 import run
    elif args.phase == 2:
        from wsr_tune.phase2 import run
    else:
        from wsr_tune.phase3 import run
    run(args)


if __name__ == "__main__":
    main()
