#!/usr/bin/env python3
"""统一 LLM→MLLM 遗忘训练框架（唯一训练入口）。

    L = 2[(1-γ)·L_obj(mm) + γ·L_obj(um)] + λ·KL(π_ref‖π_θ)

- baseline 组：`--gamma_mode static|fixed`（常数 γ，默认 0.5）
- MAW 组    ：`--gamma_mode sigmoid_ema|sigmoid_gap|ema_level`（动态 γ，需要 margin）
- 方法 = `objectives.py` 里的插件；本文件不含任何具体方法逻辑。
- 按需前向：静态 γ 且 objective 不需要 w 时，**完全不前向 w**。

用法：
    python -m exp.unlearn.framework --objective GA --gamma_mode static --run_dir results/baseline/GA/<ts>
    python -m exp.unlearn.framework --objective simPO --gamma_mode ema_level --run_dir results/MAW/simPO/<ts>
"""
import argparse
import json
import os
import time
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm

from .. import _paths
from ._common import (
    find_all_linear_names, load_model_and_processor, set_global_seed,
)
from .controller import GammaController, DYNAMIC_MODES
from .data import build_forget_loader, build_retain_loader
from .objectives import OBJECTIVES, get_objective
from .retain import compute_retain_kl


def _build_model(args, objective):
    set_global_seed(42)
    need_ref = objective.needs_ref or args.lmbda > 0
    model, ref_model, processor = load_model_and_processor(args, load_ref=need_ref)

    tokenizer = AutoTokenizer.from_pretrained(args.processor_dir, local_files_only=True)
    print("Tokenizer Length:", len(tokenizer))
    model.resize_token_embeddings(len(processor.tokenizer))
    if ref_model is not None:
        ref_model.resize_token_embeddings(len(processor.tokenizer))

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
    args.lora_dropout = lora_config.lora_dropout
    args.lora_target_modules = sorted(lora_config.target_modules)
    # args.json 需在 lora 配置确定后写入，且仅主进程写
    if os.environ.get("LOCAL_RANK", "0") == "0":
        with open(os.path.join(args.config_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    if ref_model is not None:
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()
    return model, ref_model, processor


def main(args):
    objective = get_objective(args.objective)
    controller = GammaController(args)

    model, ref_model, processor = _build_model(args, objective)

    dl_forget, n_forget = build_forget_loader(args, processor)
    dl_retain = build_retain_loader(args, processor, n_forget)

    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    writer = SummaryWriter(log_dir=args.tensorboard_dir) if accelerator.is_main_process else None
    global_step = 0

    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0,
        num_training_steps=len(dl_forget) * args.num_epochs,
    )

    to_prepare = [model, optimizer, dl_forget, dl_retain, lr_scheduler]
    if ref_model is not None:
        to_prepare.insert(1, ref_model)
    model, *prepared = accelerator.prepare(*to_prepare)
    if ref_model is not None:
        ref_model, optimizer, dl_forget, dl_retain, lr_scheduler = prepared
    else:
        optimizer, dl_forget, dl_retain, lr_scheduler = prepared

    need_w = objective.needs_w or controller.needs_margin
    print(f"== objective={args.objective} group={args.group} gamma_mode={args.gamma_mode} "
          f"needs_ref={objective.needs_ref} need_w={need_w} lambda={args.lmbda} ==")

    stop_training = False
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(zip(dl_forget, dl_retain),
                        desc=f"Epoch {epoch+1}", total=len(dl_forget))

        for pair_forget, pair_retain in progress:
            mm, um = pair_forget["mm"], pair_forget["um"]
            w_mm = mm["batch_w"] if need_w else None
            w_um = um["batch_w"] if need_w else None

            loss_mm, rl_mm, rw_mm = objective.fn(model, ref_model, mm["batch_l"], w_mm, args)
            loss_um, rl_um, rw_um = objective.fn(model, ref_model, um["batch_l"], w_um, args)

            step_log = {"loss_mm": loss_mm.item(), "loss_um": loss_um.item()}
            if controller.needs_margin:
                with torch.no_grad():
                    m_mm = accelerator.reduce((rw_mm - rl_mm).mean(), reduction="mean")
                    m_um = accelerator.reduce((rw_um - rl_um).mean(), reduction="mean")
                    gap = float(m_mm - m_um)
                gamma, info = controller.update(gap)
                step_log.update(gamma=gamma, gap=gap, M_mm=float(m_mm), M_um=float(m_um), **info)
            else:
                gamma = controller.static_value()
                gap = 0.0
                step_log["gamma"] = gamma

            loss_forget = 2.0 * ((1.0 - gamma) * loss_mm + gamma * loss_um)
            accelerator.backward(loss_forget)

            step_total = loss_forget.item()
            if args.lmbda > 0:
                ret_mm, ret_um = pair_retain["mm"], pair_retain["um"]
                loss_ret_mm = compute_retain_kl(model, ref_model, ret_mm)
                accelerator.backward(args.lmbda * loss_ret_mm)
                loss_ret_um = compute_retain_kl(model, ref_model, ret_um)
                accelerator.backward(args.lmbda * loss_ret_um)
                step_log["l_ret_mm"] = loss_ret_mm.item()
                step_log["l_ret_um"] = loss_ret_um.item()
                step_total += args.lmbda * (loss_ret_mm.item() + loss_ret_um.item())

            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            total_loss += step_total
            if writer is not None:
                writer.add_scalar("Loss/train", step_total, global_step)
                writer.add_scalar("gamma", gamma, global_step)
                writer.add_scalar("M/margin_gap", gap, global_step)
                if controller.needs_margin:
                    writer.add_scalar("M/multimodal_margin", float(m_mm), global_step)
                    writer.add_scalar("M/unimodal_margin", float(m_um), global_step)
                    writer.add_scalar("M/gap_ema", info["gap_ema"], global_step)
                    writer.add_scalar("controller/s", info["s"], global_step)
                    writer.add_scalar("controller/alpha_s", info["alpha_s"], global_step)
                    writer.add_scalar("controller/level_ema", info["level_ema"], global_step)
                    writer.add_scalar("controller/s_hat", info["s_hat"], global_step)
            global_step += 1
            progress.set_postfix({k: (f"{v:.4f}" if isinstance(v, float) else v)
                                  for k, v in step_log.items()})
            if args.max_steps is not None and global_step >= args.max_steps:
                stop_training = True
                break

        avg_loss = total_loss / len(dl_forget)
        if writer is not None:
            writer.add_scalar("Loss/epoch", avg_loss, epoch)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, gamma: {gamma:.4f}")

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            epoch_dir = os.path.join(args.epoch_dir, f"epoch-{epoch + 1}", "model")
            os.makedirs(epoch_dir, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(epoch_dir)
            print(f"Saved LoRA checkpoint: {epoch_dir}")
        if stop_training:
            break

    if writer is not None:
        writer.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_epoch_dir = os.path.join(args.epoch_dir, f"epoch-{epoch + 1}", "model")
        with open(os.path.join(final_epoch_dir, "base_model.json"), "w") as f:
            json.dump({"base_model": args.vanilla_dir, "method": args.objective,
                       "group": args.group, "gamma_mode": args.gamma_mode}, f)
        if os.path.lexists(args.save_dir):
            if os.path.islink(args.save_dir):
                os.unlink(args.save_dir)
            elif os.path.isdir(args.save_dir) and not os.listdir(args.save_dir):
                os.rmdir(args.save_dir)
            else:
                raise FileExistsError(f"Refusing to replace non-empty final adapter: {args.save_dir}")
        target = os.path.relpath(final_epoch_dir, os.path.dirname(args.save_dir))
        os.symlink(target, args.save_dir, target_is_directory=True)

        with open(os.path.join(args.config_dir, "controller_state.json"), "w") as f:
            json.dump(controller.state(), f, indent=2)

    print(f"Model saved to: {args.save_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified LLM->MLLM unlearning framework (pluggable objectives)")
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), required=True)
    parser.add_argument("--group", choices=("baseline", "maw"), default="baseline")
    parser.add_argument("--model_id", type=str, default=_paths.VANILLA)
    parser.add_argument("--processor_dir", type=str, default=_paths.ORIGIN)
    parser.add_argument("--vanilla_dir", type=str, default=_paths.ORIGIN,
                        help="SFT/origin model path (policy init; also π_ref when needed)")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--data_split_dir", type=str, default=_paths.UMU_BENCH)
    parser.add_argument("--forget_split_ratio", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    # objective hyperparams
    parser.add_argument("--beta", type=float, default=0.4, help="DPO/NPO/simPO temperature")
    parser.add_argument("--obj_gamma", type=float, default=0.0,
                        help="NPO/simPO margin threshold (distinct from controller gamma)")
    parser.add_argument("--coeff", choices=("inv_beta", "one"), default="inv_beta",
                        help="DPO forget loss coefficient: /beta or 1")
    # controller
    parser.add_argument("--gamma_mode",
                        choices=("static", "fixed", "sigmoid_ema", "sigmoid_gap", "ema_level"),
                        default="static")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="静态 γ（baseline 组，可搜）")
    parser.add_argument("--gamma_fixed", type=float, default=0.5, help="gamma_mode=fixed 的常数")
    parser.add_argument("--alpha", type=float, default=1.0, help="控制器 sigmoid 增益")
    parser.add_argument("--rho", type=float, default=0.8, help="gap EMA 平滑系数")
    parser.add_argument("--bias_correct", choices=("on", "off"), default="on")
    # retain
    parser.add_argument("--lmbda", type=float, default=1.5, help="retain KL 权重（可搜）")
    return parser


def _default_run_dir(args):
    if args.group == "baseline":
        return os.path.join("results", "baseline", args.objective)
    return os.path.join("results", "MAW", args.objective)


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.num_epochs < 1:
        raise SystemExit("--num_epochs must be at least 1")
    if args.gamma_mode in DYNAMIC_MODES and not 0 <= args.rho < 1:
        raise SystemExit("--rho must be in [0, 1) for dynamic gamma modes")
    if args.gamma_mode == "static" and not 0.0 <= args.gamma <= 1.0:
        raise SystemExit("--gamma must be in [0, 1]")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join(_default_run_dir(args), timestamp)
    config_dir = os.path.join(run_dir, "config")
    runs_dir = os.path.join(run_dir, "runs")
    save_dir = args.save_dir or os.path.join(run_dir, "model")
    tb_dir = os.path.join(run_dir, "logs", "tensorboard")
    for directory in (config_dir, runs_dir, tb_dir):
        os.makedirs(directory, exist_ok=True)

    args.run_dir = run_dir
    args.save_dir = save_dir
    args.epoch_dir = runs_dir
    args.config_dir = config_dir
    args.tensorboard_dir = tb_dir
    print(f"Run dir: {run_dir}\nModel save dir: {save_dir}\nTensorBoard: {tb_dir}")

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
