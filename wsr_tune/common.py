"""Model loading and validation of artifacts shared across the three phases."""

import hashlib
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .data import resolve_chat_format
from .modules import WSRLinear, replace_module, select_target_modules


DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
LOGGER = logging.getLogger(__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(value):
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


def prepare_output(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty. Choose a new output directory.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_config.json", vars(args))
    return output


def load_model_and_tokenizer(args):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("This release supports one process. Launch with python.")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. CPU smoke tests use --device cpu --dtype float32."
            )
        if torch.cuda.device_count() != 1:
            raise ValueError("Select one GPU with CUDA_VISIBLE_DEVICES before running.")
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bfloat16; use --dtype float32.")
    elif args.dtype != "float32":
        raise ValueError("Use --dtype float32 for CPU execution.")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must provide a pad token or an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    chat_format = resolve_chat_format(tokenizer, args.chat_format)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=DTYPES[args.dtype],
        device_map={"": args.device},
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    targets = select_target_modules(model, args.layer_types, args.target_layers)
    LOGGER.info(
        "Loaded %s; selected %d projections; format=%s.",
        args.model,
        len(targets),
        chat_format,
    )
    return model, tokenizer, targets, chat_format


def make_metadata(args, model, targets, chat_format):
    return {
        "format_version": 1,
        "stage": args.phase,
        "artifact_id": str(uuid4()),
        "model": model_identity(args.model),
        "revision": getattr(model.config, "_commit_hash", None) or args.revision,
        "dtype": args.dtype,
        "chat_format": chat_format,
        "layer_types": sorted(args.layer_types.split(",")),
        "target_layers": args.target_layers,
        "modules": {
            name: list(module.weight.shape) for name, module in targets.items()
        },
    }


def read_metadata(directory, stage):
    path = Path(directory) / "metadata.json"
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("format_version") != 1 or metadata.get("stage") != stage:
        raise ValueError(f"{path} is not a WSR-Tune Phase {stage} artifact.")
    return metadata


def validate_metadata(saved, current):
    for key in ("model", "revision", "dtype", "chat_format", "modules"):
        if saved.get(key) != current.get(key):
            raise ValueError(
                f"Artifact mismatch in {key}. Keep the model, dtype, input format, "
                "layer_types, and target_layers consistent across phases."
            )


def install_wsr_modules(model, targets, basis_dir, masks_dir=None):
    modules = {}
    for name, linear in targets.items():
        basis_data = torch.load(
            Path(basis_dir) / f"{name}.pt", map_location="cpu", weights_only=True
        )
        mask = None
        if masks_dir is not None:
            mask = torch.load(
                Path(masks_dir) / f"{name}.pt", map_location="cpu", weights_only=True
            )["mask"]
        module = WSRLinear(linear, basis_data["U"], mask)
        replace_module(model, name, module)
        modules[name] = module
    # Callers should release `targets`: its values refer to the original weights.
    return modules
