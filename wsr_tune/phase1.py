"""Phase 1: collect input activation Gram matrices and save complete bases."""

import gc
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .common import (
    file_sha256,
    load_model_and_tokenizer,
    make_metadata,
    prepare_output,
    write_json,
)
from .data import PairDataset, ResponseCollator, read_safety_pairs


LOGGER = logging.getLogger(__name__)


def activation_gram(activations, attention_mask, dtype):
    """Return H H.T in input coordinates, excluding padding positions."""
    activations = activations.to(dtype=dtype)
    mask = attention_mask.to(device=activations.device, dtype=dtype)
    flat = (activations * mask.unsqueeze(-1)).reshape(-1, activations.shape[-1])
    return flat.T @ flat


def basis_from_gram(gram):
    if not torch.isfinite(gram).all():
        raise ValueError("Non-finite Gram matrix; try --gram_dtype float32.")
    # The right singular vectors of the Gram matrix form the complete square
    # basis U; singular values are kept for inspection and never truncate it.
    _, singular_values, vh = torch.linalg.svd(gram.float(), full_matrices=False)
    return vh.T.contiguous(), singular_values


def collect_and_save_basis(
    model, loader, targets, output_dir, layers_per_pass=0, gram_dtype="model"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_indices = sorted({int(name.split(".")[2]) for name in targets})
    group_size = layers_per_pass or len(layer_indices)
    device = model.get_input_embeddings().weight.device
    model.eval()
    token_counts = {}
    for start in range(0, len(layer_indices), group_size):
        group_indices = set(layer_indices[start : start + group_size])
        group = {
            name: module
            for name, module in targets.items()
            if int(name.split(".")[2]) in group_indices
        }
        grams, handles = {}, []
        current_mask = None

        def make_hook(name):
            def hook(module, inputs):
                dtype = inputs[0].dtype if gram_dtype == "model" else torch.float32
                gram = activation_gram(inputs[0], current_mask, dtype)
                if name not in grams:
                    grams[name] = gram
                else:
                    grams[name].add_(gram)
                token_counts[name] = token_counts.get(name, 0) + int(current_mask.sum())

            return hook

        try:
            for name, module in group.items():
                handles.append(module.register_forward_pre_hook(make_hook(name)))
            with torch.no_grad():
                for batch in tqdm(
                    loader, desc=f"Basis: layers {sorted(group_indices)}"
                ):
                    current_mask = batch["attention_mask"].to(device)
                    # The decoder alone produces identical projection inputs and
                    # avoids allocating vocabulary logits during calibration.
                    model.model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=current_mask,
                        use_cache=False,
                    )
        finally:
            for handle in handles:
                handle.remove()
        for name, module in group.items():
            if name not in grams or not token_counts[name]:
                raise RuntimeError(f"No calibration activations collected for {name}.")
            basis, values = basis_from_gram(grams.pop(name))
            torch.save(
                {
                    "U": basis.to(dtype=module.weight.dtype).cpu(),
                    "S": values.cpu(),
                    "num_tokens": token_counts[name],
                },
                output_dir / f"{name}.pt",
            )
            LOGGER.info(
                "Saved %s: basis=%s, tokens=%d",
                name,
                tuple(basis.shape),
                token_counts[name],
            )
            del basis, values
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return token_counts


def run(args):
    pairs = read_safety_pairs(args.safety_data, args.safety_samples)
    output = prepare_output(args)
    model, tokenizer, targets, chat_format = load_model_and_tokenizer(args)
    dataset = PairDataset(
        pairs,
        tokenizer,
        args.max_length,
        chat_format,
        calibration=True,
        require_response=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ResponseCollator(tokenizer.pad_token_id),
    )
    basis_dir = output / "basis"
    counts = collect_and_save_basis(
        model, loader, targets, basis_dir, args.layers_per_pass, args.gram_dtype
    )
    metadata = make_metadata(args, model, targets, chat_format)
    metadata.update(
        {
            "safety_sha256": file_sha256(args.safety_data),
            "num_examples": len(dataset),
            "max_length": args.max_length,
            "gram_dtype": args.gram_dtype,
            "token_counts": counts,
        }
    )
    write_json(basis_dir / "metadata.json", metadata)
    LOGGER.info("Phase 1 complete: %s", basis_dir)
