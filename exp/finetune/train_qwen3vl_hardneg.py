"""Hard-negative likelihood-margin finetuning for Qwen3-VL on UMU-bench."""

import argparse
import json
import os
import random
import sys

if "--no-use_unsloth" not in sys.argv:
    try:
        import unsloth  # noqa: F401
    except Exception:
        pass

import torch
from transformers import Trainer, TrainingArguments, set_seed

from .. import _paths
from .qwenvl.collator import HardNegativeCollator
from .qwenvl.datasets import UmuHardNegativeDataset
from .train_qwen3vl import load_model, load_processor, resolve_optim, save_merged


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL hard-negative margin SFT")
    parser.add_argument("--model_id", default=_paths.QWEN3VL_VANILLA)
    parser.add_argument("--parquet", "--data_dir", dest="parquet", required=True)
    parser.add_argument("--output_dir", "--save_dir", dest="output_dir", required=True)
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--logging_dir", default="")
    parser.add_argument("--init_adapter", required=True)
    parser.add_argument("--modalities", choices=("mm", "um", "mm_um"), default="mm_um")
    parser.add_argument("--epochs", "--num_epochs", dest="epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--min_pixels", type=int, default=28 * 28 * 16)
    parser.add_argument("--max_pixels", type=int, default=28 * 28 * 576)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_targets",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=("auto", "sdpa", "flash_attention_2", "eager"),
        default="auto",
    )
    parser.add_argument("--use_unsloth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", choices=("no", "epoch", "steps"), default="no")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--skip_save", action="store_true")
    parser.add_argument("--deepspeed", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--hard_neg_k", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--neg_weight", type=float, default=0.5)
    parser.add_argument("--exclude_keys", default="Name,Description")
    return parser.parse_args()


class HardNegativeTrainer(Trainer):
    def __init__(self, *args, margin: float = 0.5, neg_weight: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.neg_weight = neg_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        counts = inputs.pop("candidate_counts").tolist()
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~mask, 0)
        target_logits = shift_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        log_z = torch.logsumexp(shift_logits, dim=-1)
        token_logp = (target_logits - log_z).float()
        token_counts = mask.sum(-1).clamp(min=1)
        scores = (token_logp * mask).sum(-1) / token_counts
        losses = []
        for group in torch.split(scores, counts):
            positive = group[0]
            negatives = group[1:]
            ce = -positive
            if negatives.numel():
                margin = torch.relu(self.margin - (positive - negatives)).mean()
            else:
                margin = positive.new_zeros(())
            losses.append(ce + self.neg_weight * margin)
        loss = torch.stack(losses).mean()
        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    processor = load_processor(args.model_id, args)
    dataset = UmuHardNegativeDataset(
        args.parquet,
        processor,
        modalities=args.modalities,
        max_samples=args.max_samples,
        hard_neg_k=args.hard_neg_k,
        exclude_keys=tuple(item.strip() for item in args.exclude_keys.split(",") if item.strip()),
        seed=args.seed,
    )
    if not dataset:
        raise ValueError("no hard-negative samples were constructed")

    model, used_unsloth = load_model(args.model_id, args)
    data_collator = HardNegativeCollator(processor.tokenizer, include_position_ids=True)

    checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if args.skip_save:
        args.save_strategy = "no"

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        optim=resolve_optim(),
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir or os.path.join(checkpoint_dir, "tensorboard"),
        report_to=["tensorboard"],
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        label_names=["labels"],
        gradient_checkpointing=args.gradient_checkpointing and not used_unsloth,
        seed=args.seed,
        data_seed=args.seed,
        deepspeed=args.deepspeed or None,
        prediction_loss_only=True,
    )

    trainer = HardNegativeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor.tokenizer,
        margin=args.margin,
        neg_weight=args.neg_weight,
    )
    trainer.train(resume_from_checkpoint=args.resume or None)

    if args.skip_save:
        os.makedirs(args.output_dir, exist_ok=True)
        last = trainer.state.log_history[-1] if trainer.state.log_history else {}
        benchmark = {
            "args": vars(args),
            "used_unsloth": used_unsloth,
            "global_step": trainer.state.global_step,
            "train_runtime": last.get("train_runtime"),
            "train_samples_per_second": last.get("train_samples_per_second"),
            "train_steps_per_second": last.get("train_steps_per_second"),
            "train_loss": last.get("train_loss"),
            "peak_memory_gb": (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else None
            ),
        }
        with open(os.path.join(args.output_dir, "benchmark.json"), "w", encoding="utf-8") as handle:
            json.dump(benchmark, handle, ensure_ascii=False, indent=2, default=str)
        print(json.dumps(benchmark, ensure_ascii=False, indent=2))
        return

    trainer.save_state()
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    save_merged(unwrapped, processor, args.output_dir, used_unsloth)

    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cli": vars(args),
                "trainer": training_args.to_dict(),
                "num_train_samples": len(dataset),
                "used_unsloth": used_unsloth,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"Saved hard-negative Qwen3-VL model to {args.output_dir}")


if __name__ == "__main__":
    main()
