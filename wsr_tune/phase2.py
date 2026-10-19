"""Phase 2: accumulate absolute safety gradients and create per-matrix masks."""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .common import (
    file_sha256,
    install_wsr_modules,
    load_model_and_tokenizer,
    make_metadata,
    prepare_output,
    read_metadata,
    validate_metadata,
    write_json,
)
from .data import PairDataset, ResponseCollator, read_safety_pairs


LOGGER = logging.getLogger(__name__)


def importance_mask(scores, keep_ratio):
    """Protect the top keep_ratio fraction per matrix; ties at the threshold are kept."""
    if not 0 <= keep_ratio <= 1:
        raise ValueError("keep_ratio must be between 0 and 1.")
    values = scores.detach().float().cpu().numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Importance scores must be finite and nonnegative.")
    if keep_ratio == 0:
        return torch.zeros_like(scores, dtype=torch.bool, device="cpu")
    if keep_ratio == 1:
        return torch.ones_like(scores, dtype=torch.bool, device="cpu")
    threshold = np.quantile(values, 1 - keep_ratio)
    return torch.from_numpy(values >= threshold)


def score_importance(model, modules, loader):
    model.eval()
    model.requires_grad_(False)
    for module in modules.values():
        module.basis_coeff.requires_grad_(True)
    device = model.get_input_embeddings().weight.device
    scores = {}
    batches = 0
    for batch in tqdm(loader, desc="Safety importance"):
        batch = {key: value.to(device) for key, value in batch.items()}
        if not (batch["labels"][:, 1:] != -100).any():
            continue
        model.zero_grad(set_to_none=True)
        result = model(**batch, use_cache=False)
        if not torch.isfinite(result.loss):
            raise RuntimeError("Non-finite safety loss; check data and precision.")
        result.loss.backward()
        for name, module in modules.items():
            grad = module.basis_coeff.grad
            if grad is None or not torch.isfinite(grad).all():
                raise RuntimeError(f"Missing or non-finite importance gradient: {name}")
            # Sum per-batch absolute gradients in the model dtype.
            if name not in scores:
                scores[name] = grad.detach().abs().clone()
            else:
                scores[name].add_(grad.detach().abs())
        batches += 1
        model.zero_grad(set_to_none=True)
        del result
    if not batches:
        raise ValueError("No batches with supervised response tokens.")
    return scores


def run(args):
    basis_metadata = read_metadata(args.basis_dir, stage=1)
    pairs = read_safety_pairs(args.safety_data, args.safety_samples)
    output = prepare_output(args)
    model, tokenizer, targets, chat_format = load_model_and_tokenizer(args)
    metadata = make_metadata(args, model, targets, chat_format)
    validate_metadata(basis_metadata, metadata)
    modules = install_wsr_modules(model, targets, args.basis_dir)
    del targets
    dataset = PairDataset(
        pairs, tokenizer, args.max_length, chat_format, calibration=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ResponseCollator(tokenizer.pad_token_id),
    )
    scores = score_importance(model, modules, loader)
    masks_dir, scores_dir = output / "masks", output / "importance"
    masks_dir.mkdir()
    scores_dir.mkdir()
    frozen_counts = {}
    for name in list(scores):
        score = scores.pop(name).float().cpu()
        mask = importance_mask(score, args.keep_ratio)
        torch.save({"importance": score}, scores_dir / f"{name}.pt")
        torch.save({"mask": mask}, masks_dir / f"{name}.pt")
        frozen_counts[name] = int(mask.sum())
        LOGGER.info(
            "%s: protected %d/%d coefficients (%.3f%%)",
            name,
            frozen_counts[name],
            mask.numel(),
            100 * frozen_counts[name] / mask.numel(),
        )
    metadata.update(
        {
            "basis_id": basis_metadata["artifact_id"],
            "keep_ratio": args.keep_ratio,
            "safety_sha256": file_sha256(args.safety_data),
            "num_examples": len(dataset),
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "frozen_counts": frozen_counts,
        }
    )
    write_json(masks_dir / "metadata.json", metadata)
    write_json(scores_dir / "metadata.json", metadata)
    LOGGER.info("Phase 2 complete: %s", masks_dir)
