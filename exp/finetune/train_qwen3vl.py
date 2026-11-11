"""Qwen3-VL supervised finetuning for the UMU-bench origin baseline.

Data processing, assistant-only loss, and mRoPE position ids follow
QwenLM/Qwen3-VL qwen-vl-finetune (Apache-2.0), while the training entry uses
the standard HF Trainer and this repo's parquet dataset layout.
"""

import argparse
import inspect
import json
import os
import random

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .. import _paths
from .qwenvl.collator import DataCollatorForSupervisedDataset
from .qwenvl.datasets import UmuQwenDataset
from .qwenvl.processor import update_processor_pixels
from .qwenvl.runtime import resolve_attn_implementation


def resolve_optim() -> str:
    if not torch.cuda.is_available():
        return "adamw_torch"
    try:
        if "fused" in inspect.signature(torch.optim.AdamW).parameters:
            return "adamw_torch_fused"
    except Exception:
        pass
    return "adamw_torch"


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-VL SFT for UMU-bench origin")
    parser.add_argument("--model_id", default=_paths.QWEN3VL_VANILLA)
    parser.add_argument("--parquet", "--data_dir", dest="parquet", required=True)
    parser.add_argument("--output_dir", "--save_dir", dest="output_dir", required=True)
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--logging_dir", default="")
    parser.add_argument("--modalities", choices=("mm", "um", "mm_um"), default="mm")
    parser.add_argument("--epochs", "--num_epochs", dest="epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    parser.add_argument("--use_unsloth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", choices=("no", "epoch", "steps"), default="no")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", default="")
    parser.add_argument("--debug_labels", action="store_true")
    return parser.parse_args()


def load_processor(model_id, args):
    processor = AutoProcessor.from_pretrained(
        model_id, local_files_only=os.path.isdir(model_id)
    )
    update_processor_pixels(processor, args.min_pixels, args.max_pixels)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_length
    return processor


def _load_standard(model_id, args):
    attn_implementation = resolve_attn_implementation(args.attn_implementation)
    load_kwargs = {
        "torch_dtype": torch.bfloat16 if args.bf16 else None,
        "local_files_only": os.path.isdir(model_id),
    }
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, attn_implementation=attn_implementation, **load_kwargs
        )
    except Exception as exc:
        if attn_implementation != "flash_attention_2":
            raise
        print(f"!! flash_attention_2 failed ({exc}); falling back to sdpa")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, attn_implementation="sdpa", **load_kwargs
        )
    print(f"Loaded Qwen3-VL with attn_implementation={getattr(model.config, '_attn_implementation', attn_implementation)}")
    model.config.use_cache = False

    for parameter in model.parameters():
        parameter.requires_grad = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[item.strip() for item in args.lora_targets.split(",") if item.strip()],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
    return model


def _load_unsloth(model_id, args):
    try:
        from unsloth import FastVisionModel
    except Exception as exc:
        print(f"Unsloth not available ({exc}); using standard HF model")
        return None
    try:
        model, _ = FastVisionModel.from_pretrained(
            model_id,
            load_in_4bit=False,
            use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
            max_seq_length=args.max_length,
        )
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
            use_gradient_checkpointing="unsloth" if args.gradient_checkpointing else False,
        )
        model.config.use_cache = False
        FastVisionModel.for_training(model)
        for name, parameter in model.named_parameters():
            if "merger" in name:
                parameter.requires_grad = False
        model.print_trainable_parameters()
        print("Loaded Qwen3-VL with Unsloth")
        return model
    except Exception as exc:
        print(f"Unsloth load failed ({exc}); using standard HF model")
        return None


def load_model(model_id, args):
    if args.use_unsloth:
        model = _load_unsloth(model_id, args)
        if model is not None:
            return model, True
    return _load_standard(model_id, args), False


def save_merged(model, processor, output_dir, used_unsloth=False):
    os.makedirs(output_dir, exist_ok=True)
    if used_unsloth and hasattr(model, "save_pretrained_merged"):
        try:
            model.config.use_cache = True
            model.save_pretrained_merged(
                output_dir, processor.tokenizer, save_method="merged_16bit"
            )
            processor.save_pretrained(output_dir)
            return
        except Exception as exc:
            print(f"!! Unsloth merged save failed ({exc}); falling back to HF merge")
    model = model.merge_and_unload() if isinstance(model, PeftModel) else model
    model.config.use_cache = True
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(output_dir)


def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)

    processor = load_processor(args.model_id, args)

    train_dataset = UmuQwenDataset(
        args.parquet,
        processor,
        modalities=args.modalities,
        max_samples=args.max_samples,
    )

    if args.debug_labels:
        sample = train_dataset[0]
        print("INPUT:", processor.tokenizer.decode(sample["input_ids"][0], skip_special_tokens=False))
        kept = [token for token in sample["labels"][0].tolist() if token != -100]
        print("LABELS:", processor.tokenizer.decode(kept, skip_special_tokens=False))
        print("POSITION_IDS_SHAPE:", tuple(sample["position_ids"].shape))
        return

    model, used_unsloth = load_model(args.model_id, args)
    data_collator = DataCollatorForSupervisedDataset(
        processor.tokenizer, include_position_ids=not used_unsloth
    )

    checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
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
        gradient_checkpointing=args.gradient_checkpointing and not used_unsloth,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor.tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_state()

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    save_merged(unwrapped, processor, args.output_dir, used_unsloth)

    with open(os.path.join(args.output_dir, "train_args.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cli": vars(args),
                "trainer": training_args.to_dict(),
                "num_train_samples": len(train_dataset),
                "used_unsloth": used_unsloth,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"Saved merged Qwen3-VL origin to {args.output_dir}")


if __name__ == "__main__":
    main()
