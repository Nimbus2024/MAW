#!/usr/bin/env python3
"""行为梯度投影: 把 unlearn 的 LoRA ΔW 投影到"行为相关"子空间。

行为 B ∈ {it, pt, gen}:
  it  = forget 实体的 image+text QA 答案 CE
  pt  = forget 实体的 text-only QA 答案 CE
  gen = retain 实体 QA 答案 CE (通用能力代理)

输出 (每个 adapter):
  - cos(ΔW, G_B)、沿 G_B 的能量占比 (cos^2)、在 [G_it,G_pt,G_gen] 子空间内的能量/残差;
  - 一阶预测 <G_B, ΔW> vs 真实行为损失变化 ΔL_B = L_B(unlearned) - L_B(base)。
"""
from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..unlearn._paired import build_pairs, collate_plain
from .adapter_geometry import load_adapter_ab

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MODALITY = {"it": "mm", "pt": "um", "gen": "mm"}
BEHAVIORS = ("it", "pt", "gen")


def norm_name(name: str) -> str:
    return name[name.index("language_model."):] if "language_model." in name else name


def target_modules(model):
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in TARGET_SUFFIXES:
            out[norm_name(name)] = module
    return out


def probe_batches(data_split_dir: str, split: str, n_batches: int, batch_size: int,
                  processor, max_length: int, seed: int = 42):
    pairs = build_pairs(pd.read_parquet(Path(data_split_dir) / split / "train-00000-of-00001.parquet"))
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed)).tolist()
    pairs = [pairs[i] for i in order]
    args = types.SimpleNamespace(max_length=max_length)
    batches = []
    for b in range(n_batches):
        chunk = pairs[b * batch_size:(b + 1) * batch_size]
        if len(chunk) < batch_size:
            break
        items = [{"mm": {"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]},
                  "um": {"question": p["um_q"], "answer": p["um_a"]}} for p in chunk]
        batches.append(collate_plain(items, processor, args))
    return batches


def to_device(batch, modality, device):
    mm, um = batch["mm"], batch["um"]
    if modality == "mm":
        ids, attn, pixel, labels = mm
    else:
        ids, attn, _, labels = um
        pixel = None
    ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
    if pixel is not None:
        pixel = pixel.to(device)
    return ids, attn, pixel, labels


def behavior_gradient(model, modules, batches, modality, device):
    for module in modules.values():
        module.weight.grad = None
    n = 0
    for batch in batches:
        ids, attn, pixel, labels = to_device(batch, modality, device)
        model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels).loss.backward()
        n += 1
    grads = {}
    for name, module in modules.items():
        g = module.weight.grad
        grads[name] = (g.detach() / max(n, 1)).to(torch.bfloat16).contiguous() if g is not None \
            else torch.zeros_like(module.weight, dtype=torch.bfloat16)
    for module in modules.values():
        module.weight.grad = None
    return grads


@torch.no_grad()
def behavior_loss(model, batches, modality, device):
    total, n = 0.0, 0
    for batch in batches:
        ids, attn, pixel, labels = to_device(batch, modality, device)
        loss = model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels).loss
        total += float(loss)
        n += 1
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="行为梯度投影 (ΔW -> 行为子空间)")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", required=True, help="name=path,...")
    parser.add_argument("--data_split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--forget_split", default="forget_5")
    parser.add_argument("--retain_split", default="retain_95")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device)
    model.eval()
    model.config.use_cache = False
    modules = target_modules(model)
    for module in modules.values():
        module.weight.requires_grad_(True)
    print(f"target modules: {len(modules)}", flush=True)

    probe = {
        "it": probe_batches(args.data_split_dir, args.forget_split, args.n_batches,
                            args.batch_size, processor, args.max_length, args.seed),
        "pt": probe_batches(args.data_split_dir, args.forget_split, args.n_batches,
                            args.batch_size, processor, args.max_length, args.seed),
        "gen": probe_batches(args.data_split_dir, args.retain_split, args.n_batches,
                             args.batch_size, processor, args.max_length, args.seed),
    }
    grads, base_loss = {}, {}
    for b in BEHAVIORS:
        base_loss[b] = behavior_loss(model, probe[b], MODALITY[b], args.device)
        grads[b] = behavior_gradient(model, modules, probe[b], MODALITY[b], args.device)
        print(f"computed G_{b} (L_base={base_loss[b]:.4f})", flush=True)
    for module in modules.values():
        module.weight.requires_grad_(False)

    adapter_list = [it.split("=", 1) for it in args.adapters.split(",")]
    peft = PeftModel.from_pretrained(model, adapter_list[0][1], adapter_name=adapter_list[0][0])
    peft.eval()
    unlearned_loss = {}
    for name, path in adapter_list:
        if name != adapter_list[0][0]:
            peft.load_adapter(path, adapter_name=name)
        peft.set_adapter(name)
        unlearned_loss[name] = {b: behavior_loss(peft, probe[b], MODALITY[b], args.device)
                                for b in BEHAVIORS}
        print(f"{name} L_unlearned={unlearned_loss[name]}", flush=True)

    keys = sorted(modules)
    result = {"keys": keys, "base_loss": base_loss, "adapters": {}}
    for name, path in adapter_list:
        ab, scaling = load_adapter_ab(Path(path))
        ab = {norm_name(k): v for k, v in ab.items()}
        common = [k for k in keys if k in ab]
        norm_d2 = 0.0
        norm_g2 = {b: 0.0 for b in BEHAVIORS}
        dot = {b: 0.0 for b in BEHAVIORS}
        gram = {(i, j): 0.0 for i in BEHAVIORS for j in BEHAVIORS}
        for m in common:
            A, B = ab[m]
            dW = (scaling * (B @ A)).to(args.device)
            norm_d2 += float((dW * dW).sum())
            gs = {b: grads[b][m].to(args.device).float() for b in BEHAVIORS}
            for b in BEHAVIORS:
                dot[b] += float((gs[b] * dW).sum())
                norm_g2[b] += float((gs[b] * gs[b]).sum())
            for i in BEHAVIORS:
                for j in BEHAVIORS:
                    gram[(i, j)] += float((gs[i] * gs[j]).sum())
            del dW, gs
        m_mat = torch.tensor([[gram[(i, j)] for j in BEHAVIORS] for i in BEHAVIORS], dtype=torch.float64)
        b_vec = torch.tensor([dot[b] for b in BEHAVIORS], dtype=torch.float64)
        try:
            coeffs = torch.linalg.solve(m_mat, b_vec)
            proj_energy = float(b_vec @ coeffs)
        except Exception:
            coeffs = torch.zeros(3, dtype=torch.float64)
            proj_energy = 0.0
        nd = math.sqrt(norm_d2) if norm_d2 > 0 else 0.0
        per_behavior = {}
        for b in BEHAVIORS:
            ng = math.sqrt(norm_g2[b])
            per_behavior[b] = {
                "norm_g": ng,
                "cos": dot[b] / (ng * nd) if ng > 0 and nd > 0 else None,
                "energy_frac": (dot[b] ** 2) / (norm_g2[b] * norm_d2) if ng > 0 and nd > 0 else None,
                "first_order_dL": dot[b],
                "actual_dL": unlearned_loss[name][b] - base_loss[b],
            }
        result["adapters"][name] = {
            "norm": nd,
            "n_modules": len(common),
            "per_behavior": per_behavior,
            "subspace_coeffs": {b: float(x) for b, x in zip(BEHAVIORS, coeffs)},
            "subspace_energy_frac": proj_energy / norm_d2 if norm_d2 > 0 else None,
            "residual_frac": 1 - proj_energy / norm_d2 if norm_d2 > 0 else None,
        }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'adapter':>8} {'norm':>7} {'b':>4} {'cos':>8} {'Efrac':>8} {'1st_dL':>9} {'actual_dL':>10}")
    for name, v in result["adapters"].items():
        for b in BEHAVIORS:
            pb = v["per_behavior"][b]
            print(f"{name:>8} {v['norm']:>7.3f} {b:>4} {pb['cos']:>8.4f} {pb['energy_frac']:>8.5f} "
                  f"{pb['first_order_dL']:>9.4f} {pb['actual_dL']:>10.4f}")
        print(f"{'':>8} {'':>7} {'sub':>4} Efrac={v['subspace_energy_frac']:.5f} resid={v['residual_frac']:.5f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
