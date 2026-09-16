#!/usr/bin/env python3
"""行为梯度投影: 把 unlearn 的 LoRA ΔW 投影到"行为相关"子空间。

行为 B ∈ {it, pt, gen}:
  it  = forget 实体的 image+text QA 答案 CE
  pt  = forget 实体的 text-only QA 答案 CE
  gen = retain 实体 QA 答案 CE (通用能力代理)

对每个行为算权重空间梯度 G_B = dL_B/dW (base 的 LoRA 目标 Linear 权重)，
再把每个 adapter 的 ΔW 投影到 {G_it, G_pt, G_gen}: 报告 cos、沿单方向的能量占比、
以及在 [G_it,G_pt,G_gen] 子空间内的分解能量与残差。
"""
from __future__ import annotations

import argparse
import json
import math
import re
import types
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..unlearn._paired import build_pairs, collate_plain
from .adapter_geometry import load_adapter_ab

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def norm_name(name: str) -> str:
    return name[name.index("language_model."):] if "language_model." in name else name


def load_base(base: str, device: str):
    model = LlavaForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return model


def target_params(model):
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in TARGET_SUFFIXES:
            out[norm_name(name)] = module
    return out


def probe_batches(data_split_dir: str, split: str, n_batches: int, batch_size: int,
                  processor, max_length: int, seed: int = 42):
    path = Path(data_split_dir) / split / "train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    pairs = build_pairs(df)
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pairs), generator=rng).tolist()
    pairs = [pairs[i] for i in order]
    args = types.SimpleNamespace(max_length=max_length)
    batches = []
    for b in range(n_batches):
        chunk = pairs[b * batch_size:(b + 1) * batch_size]
        if len(chunk) < batch_size:
            break
        mm = [{"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]} for p in chunk]
        um = [{"question": p["um_q"], "answer": p["um_a"]} for p in chunk]
        batches.append(collate_plain([{"mm": m, "um": u} for m, u in zip(mm, um)],
                                     processor, args))
    return batches


def zero_grads(modules):
    for module in modules.values():
        module.weight.grad = None


def behavior_gradient(model, modules, batches, modality: str, device: str):
    zero_grads(modules)
    n = 0
    for batch in batches:
        mm, um = batch["mm"], batch["um"]
        if modality == "it":
            ids, attn, pixel, labels = mm
        elif modality == "pt":
            ids, attn, _, labels = um
            pixel = None
        else:
            ids, attn, pixel, labels = mm
        ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
        if pixel is not None:
            pixel = pixel.to(device)
        out = model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels)
        out.loss.backward()
        n += 1
    grads = {}
    for name, module in modules.items():
        if module.weight.grad is None:
            grads[name] = torch.zeros_like(module.weight, dtype=torch.float32, device="cpu")
        else:
            grads[name] = (module.weight.grad.detach().float() / max(n, 1)).cpu()
    zero_grads(modules)
    return grads


def flatten_join(tensors: dict, keys):
    return torch.cat([tensors[k].flatten().double() for k in keys])


def main():
    parser = argparse.ArgumentParser(description="行为梯度投影 (ΔW -> 行为子空间)")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", required=True,
                        help="逗号分隔 name=path, 如 mm=.../model,um=.../model,joint=.../model")
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
    model = load_base(args.base, args.device)
    modules = target_params(model)
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
    grads = {}
    for behavior, modality in (("it", "it"), ("pt", "pt"), ("gen", "mm")):
        grads[behavior] = behavior_gradient(model, modules, probe[behavior], modality, args.device)
        print(f"computed G_{behavior}", flush=True)

    for b in grads:
        grads[b] = {k: v.to(torch.bfloat16) for k, v in grads[b].items()}
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    behaviors = ("it", "pt", "gen")
    keys = sorted(set(modules) & set(grads["it"]))
    result = {"keys": keys, "adapters": {}}
    for item in args.adapters.split(","):
        aname, path = item.split("=", 1)
        ab, scaling = load_adapter_ab(Path(path))
        ab = {norm_name(k): v for k, v in ab.items()}
        common = [k for k in keys if k in ab]
        norm_d2 = 0.0
        norm_g2 = {b: 0.0 for b in behaviors}
        dot = {b: 0.0 for b in behaviors}
        gram = {(i, j): 0.0 for i in behaviors for j in behaviors}
        for m in common:
            A, B = ab[m]
            dW = scaling * (B @ A)
            norm_d2 += float((dW * dW).sum())
            gs = {b: grads[b][m].float() for b in behaviors}
            for b in behaviors:
                dot[b] += float((gs[b] * dW).sum())
                norm_g2[b] += float((gs[b] * gs[b]).sum())
            for i in behaviors:
                for j in behaviors:
                    gram[(i, j)] += float((gs[i] * gs[j]).sum())
        m_mat = torch.tensor([[gram[(i, j)] for j in behaviors] for i in behaviors],
                             dtype=torch.float64)
        b_vec = torch.tensor([dot[b] for b in behaviors], dtype=torch.float64)
        try:
            coeffs = torch.linalg.solve(m_mat, b_vec)
            proj_energy = float(b_vec @ coeffs)
        except Exception:
            coeffs = torch.zeros(3, dtype=torch.float64)
            proj_energy = 0.0
        nd = math.sqrt(norm_d2) if norm_d2 > 0 else 0.0
        per_behavior = {}
        for b in behaviors:
            ng = math.sqrt(norm_g2[b])
            per_behavior[b] = {
                "cos": dot[b] / (ng * nd) if ng > 0 and nd > 0 else None,
                "energy_frac": (dot[b] ** 2) / (norm_g2[b] * norm_d2) if ng > 0 and nd > 0 else None,
            }
        result["adapters"][aname] = {
            "norm": nd,
            "n_modules": len(common),
            "per_behavior": per_behavior,
            "subspace_coeffs": {b: float(x) for b, x in zip(behaviors, coeffs)},
            "subspace_energy_frac": proj_energy / norm_d2 if norm_d2 > 0 else None,
            "residual_frac": 1 - proj_energy / norm_d2 if norm_d2 > 0 else None,
        }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'adapter':>10} {'norm':>8} {'cos_it':>8} {'cos_pt':>8} {'cos_gen':>8} "
          f"{'Efrac_it':>9} {'Efrac_pt':>9} {'Efrac_gen':>10} {'sub_Efrac':>10} {'resid':>7}")
    for aname, v in result["adapters"].items():
        pb = v["per_behavior"]
        def f(x):
            return "NA" if x is None else f"{x:.3f}"
        print(f"{aname:>10} {v['norm']:>8.3f} {f(pb['it']['cos']):>8} {f(pb['pt']['cos']):>8} "
              f"{f(pb['gen']['cos']):>8} {f(pb['it']['energy_frac']):>9} "
              f"{f(pb['pt']['energy_frac']):>9} {f(pb['gen']['energy_frac']):>10} "
              f"{f(v['subspace_energy_frac']):>10} {f(v['residual_frac']):>7}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
