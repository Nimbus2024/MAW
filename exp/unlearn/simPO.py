#!/usr/bin/env python3
"""simPO.py — Simplified PO / SimPO-style unlearning (无参考模型, 长度归一化)。

DPO 对 (x, y_idk=拒绝回答, y_forget=原答案), 在 forget 多模态 ∪ 单模态上:
    r_idk    = (1/|y_idk|)    * Σ log π_θ(y_idk | x)
    r_forget = (1/|y_forget|) * Σ log π_θ(y_forget | x)     (仅答案 token)
    L = -E[ (1/β) * log σ( β·(r_idk - r_forget) - γ ) ]

单模型: policy = origin(SFT/llava_smu_ft) + LoRA; 无参考模型, 无 retain 项。
逐 epoch 保存到 <run>/runs/<epoch>/model, 最终 <run>/model 为末 epoch 软链。
"""
import os
import sys
import json
import random
import argparse
import time
from datetime import datetime

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoProcessor, LlavaForConditionalGeneration, get_scheduler
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .. import _paths
from .MAW import (
    ForgetDataset,
    collate_forget_mm,
    collate_forget_um,
    find_all_linear_names,
    load_idk,
    _sequence_logprob,
    set_global_seed,
    worker_init_fn,
)


def load_model_and_processor(args):
    if not _paths.is_llava(args.model_id):
        raise ValueError("仅支持 LLaVA 族模型")
    load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                       local_files_only=True)
    if "LOCAL_RANK" in os.environ:
        load_kwargs["device_map"] = {"": int(os.environ["LOCAL_RANK"])}
    else:
        load_kwargs["device_map"] = "auto"
    model = LlavaForConditionalGeneration.from_pretrained(args.vanilla_dir, **load_kwargs)
    proc_dir = args.processor_dir if args.processor_dir else args.model_id
    processor = AutoProcessor.from_pretrained(proc_dir, local_files_only=True)
    processor.num_additional_image_tokens = 1
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    return model, processor


def _forward(model, batch):
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch.get("pixel_values"),
    )


def compute_simpo_loss(model, batch_w, batch_l, beta, gamma):
    out_w = _forward(model, batch_w)
    out_l = _forward(model, batch_l)
    r_idk = _sequence_logprob(out_w.logits, batch_w["labels"], normalize=True)
    r_forget = _sequence_logprob(out_l.logits, batch_l["labels"], normalize=True)
    return -(1.0 / beta) * F.logsigmoid(beta * (r_idk - r_forget) - gamma)


def main(args):
    set_global_seed(42)
    model, processor = load_model_and_processor(args)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=find_all_linear_names(model), init_lora_weights="gaussian")
    args.lora_dropout = lora_config.lora_dropout
    args.lora_target_modules = sorted(lora_config.target_modules)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if os.environ.get("LOCAL_RANK", "0") == "0":
        with open(os.path.join(args.config_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, default=str)

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    df_forget = pd.read_parquet(
        os.path.join(forget_folder, "train-00000-of-00001.parquet"))

    idk_list = load_idk()
    print(f"Loaded {len(idk_list)} IDK responses")
    forget_mm = ForgetDataset(df_forget, idk_list, multimodal=True)
    forget_um = ForgetDataset(df_forget, idk_list, multimodal=False)
    print(f"Forget MM: {len(forget_mm)}, Forget UM: {len(forget_um)}")

    dl_mm = DataLoader(forget_mm, batch_size=args.batch_size, shuffle=True,
                       collate_fn=lambda x: collate_forget_mm(x, processor, args))
    dl_um = DataLoader(forget_um, batch_size=args.batch_size, shuffle=True,
                       collate_fn=lambda x: collate_forget_um(x, processor, args))

    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    writer = SummaryWriter(log_dir=args.tensorboard_dir) if accelerator.is_main_process else None
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=len(dl_mm) * args.num_epochs)
    model, optimizer, dl_mm, dl_um, lr_scheduler = accelerator.prepare(
        model, optimizer, dl_mm, dl_um, lr_scheduler)

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        bar = tqdm(zip(dl_mm, dl_um), desc=f"Epoch {epoch+1}", total=len(dl_mm))
        for (mm, um) in bar:
            loss_mm = compute_simpo_loss(
                model, mm["batch_w"], mm["batch_l"], args.beta, args.gamma)
            loss_um = compute_simpo_loss(
                model, um["batch_w"], um["batch_l"], args.beta, args.gamma)
            loss = (loss_mm.mean() + loss_um.mean()) / 2.0
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            total_loss += loss.item()
            if writer is not None:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("Loss/mm", loss_mm.mean().item(), global_step)
                writer.add_scalar("Loss/um", loss_um.mean().item(), global_step)
            global_step += 1
            bar.set_postfix(loss=loss.item())
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        avg_loss = total_loss / len(dl_mm)
        if writer is not None:
            writer.add_scalar("Loss/epoch", avg_loss, epoch)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            epoch_dir = os.path.join(args.epoch_dir, f"epoch-{epoch + 1}", "model")
            os.makedirs(epoch_dir, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(epoch_dir)
            print(f"Saved LoRA checkpoint: {epoch_dir}")
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    if writer is not None:
        writer.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_epoch_dir = os.path.join(args.epoch_dir, f"epoch-{epoch + 1}", "model")
        with open(os.path.join(final_epoch_dir, "base_model.json"), "w") as f:
            json.dump({"base_model": args.vanilla_dir, "method": "simPO"}, f)
        if os.path.lexists(args.save_dir):
            if os.path.islink(args.save_dir):
                os.unlink(args.save_dir)
            elif os.path.isdir(args.save_dir) and not os.listdir(args.save_dir):
                os.rmdir(args.save_dir)
            else:
                raise FileExistsError(f"Refusing to replace non-empty final adapter: {args.save_dir}")
        target = os.path.relpath(final_epoch_dir, os.path.dirname(args.save_dir))
        os.symlink(target, args.save_dir, target_is_directory=True)
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simPO: Simplified SimPO unlearning")
    parser.add_argument("--model_id", type=str, default=_paths.VANILLA)
    parser.add_argument("--processor_dir", type=str, default=_paths.ORIGIN)
    parser.add_argument("--vanilla_dir", type=str, default=_paths.ORIGIN,
                        help="policy 起点模型(origin/SFT), 本地路径")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--data_split_dir", type=str, default=_paths.UMU_BENCH)
    parser.add_argument("--forget_split_ratio", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join("results", "simPO", timestamp)
    config_dir = os.path.join(run_dir, "config")
    runs_dir = os.path.join(run_dir, "runs")
    save_dir = os.path.join(run_dir, "model")
    tb_dir = os.path.join(run_dir, "logs", "tensorboard")
    for directory in (config_dir, runs_dir, tb_dir):
        os.makedirs(directory, exist_ok=True)
    args.run_dir = run_dir
    args.save_dir = save_dir
    args.epoch_dir = runs_dir
    args.config_dir = config_dir
    args.tensorboard_dir = tb_dir

    try:
        main(args)
    except Exception:
        import traceback
        crash = {"timestamp": datetime.now().strftime("%y%m%d_%H%M%S"),
                 "traceback": traceback.format_exc()}
        with open(os.path.join(config_dir, "crash_report.json"), "w") as f:
            json.dump(crash, f, indent=2)
        print(f"Crash report saved to {os.path.join(config_dir, 'crash_report.json')}")
        raise
