"""Phase 3: downstream fine-tuning while keeping protected coefficients fixed."""

import gc
import logging

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from .common import (
    install_wsr_modules,
    load_model_and_tokenizer,
    make_metadata,
    prepare_output,
    read_metadata,
    validate_metadata,
    write_json,
)
from .data import TASKS, PairDataset, ResponseCollator, read_task_examples
from .modules import fold_wsr_modules


LOGGER = logging.getLogger(__name__)


class RestoreFrozenCallback(TrainerCallback):
    def __init__(self, modules):
        self.modules = modules

    def on_step_end(self, args, state, control, **kwargs):
        # Gradient masking alone does not block AdamW's decoupled weight decay.
        for module in self.modules.values():
            module.restore_frozen()
        return control


def run(args):
    basis_metadata = read_metadata(args.basis_dir, stage=1)
    mask_metadata = read_metadata(args.masks_dir, stage=2)
    if mask_metadata.get("basis_id") != basis_metadata["artifact_id"]:
        raise ValueError("The masks were generated from a different Phase 1 basis.")
    output = prepare_output(args)
    model, tokenizer, targets, chat_format = load_model_and_tokenizer(args)
    metadata = make_metadata(args, model, targets, chat_format)
    validate_metadata(basis_metadata, metadata)
    validate_metadata(mask_metadata, metadata)
    modules = install_wsr_modules(model, targets, args.basis_dir, args.masks_dir)
    del targets
    model.requires_grad_(True)
    examples = read_task_examples(
        args.task, args.task_path, args.task_samples, seed=args.seed
    )
    dataset = PairDataset(examples, tokenizer, args.max_length, chat_format)
    protected = sum(int(module.coeff_mask.sum()) for module in modules.values())
    LOGGER.info(
        "%s examples: %d; protected coefficients: %d",
        TASKS[args.task].name,
        len(dataset),
        protected,
    )
    training_args = TrainingArguments(
        output_dir=str(output / "trainer"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        optim="adamw_torch",
        bf16=args.dtype == "bfloat16",
        use_cpu=args.device == "cpu",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        dataloader_pin_memory=args.device == "cuda",
        seed=args.seed,
        data_seed=args.seed,
    )
    callback = RestoreFrozenCallback(modules)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=ResponseCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
        callbacks=[callback],
    )
    result = trainer.train()
    for module in modules.values():
        module.assert_frozen()
    metadata.update(
        {
            "basis_id": basis_metadata["artifact_id"],
            "mask_id": mask_metadata["artifact_id"],
            "task": args.task,
            "task_source": args.task_path or TASKS[args.task].source,
            "num_examples": len(dataset),
            "protected_coefficients": protected,
            "protected_coefficients_unchanged": True,
            "optimizer_steps": trainer.state.global_step,
            "metrics": result.metrics,
        }
    )
    # Release optimizer state and wrapper references before folding large weights.
    del trainer, callback, modules, module
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    fold_wsr_modules(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    final_dir = output / "final_model"
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    write_json(output / "training_summary.json", metadata)
    LOGGER.info("Phase 3 complete. Standard Hugging Face model: %s", final_dir)
