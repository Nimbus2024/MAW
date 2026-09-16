#!/usr/bin/env python3
"""对角 Fisher 归因 (PerTA 思想): 逐参数 f=E[g^2] 分别在 forget / retain 上估计,
再看各 adapter 的 ΔW 落在"遗忘敏感"还是"保留敏感"参数上。

输出 per-module / global 的 Σ ΔW²·f_forget 与 Σ ΔW^2·f_retain 及其比值。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..unlearn._paired import build_pairs
from .adapter_geometry import load_adapter_ab
from .patching import batch_to_device, build_batch, norm_name, target_modules


def pairs_of(data_split_dir, split, n, seed=42):
    ps = build_pairs(pd.read_parquet(Path(data_split_dir) / split / "train-00000-of-00001.parquet"))
    order = torch.randperm(len(ps), generator=torch.Generator().manual_seed(seed)).tolist()
    return [ps[i] for i in order[:n]]


def fisher(model, modules, pairs, processor, max_length, device, batch_size):
    for m in modules.values():
        m.weight.grad = None
    n = 0
    for b in range(0, len(pairs), batch_size):
        chunk = pairs[b:b + batch_size]
        for modality in ("mm", "um"):
            items = [{"image": p["image"] if modality == "mm" else None,
                      "question": p["mm_q"] if modality == "mm" else p["um_q"],
                      "answer": p["mm_a"] if modality == "mm" else p["um_a"]} for p in chunk]
            batch = build_batch(items, modality, processor, max_length)
            ids, attn, pixel, labels = batch_to_device(batch, modality, device)
            model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels).loss.backward()
            n += 1
    f = {}
    for name, m in modules.items():
        g = m.weight.grad
        f[name] = (g.detach().float() ** 2 / max(n, 1)).cpu() if g is not None else torch.zeros_like(m.weight, dtype=torch.float32)
        m.weight.grad = None
    return f


def main():
    parser = argparse.ArgumentParser(description="对角 Fisher 归因")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", required=True, help="name=path,...")
    parser.add_argument("--data_split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_forget", type=int, default=25)
    parser.add_argument("--n_retain", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
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
    for m in modules.values():
        m.weight.requires_grad_(True)
    f_forget = fisher(model, modules, pairs_of(args.data_split_dir, args.forget_split, args.n_forget, args.seed),
                      processor, args.max_length, args.device, args.batch_size)
    print("forget fisher done", flush=True)
    f_retain = fisher(model, modules, pairs_of(args.data_split_dir, args.retain_split, args.n_retain, args.seed),
                      processor, args.max_length, args.device, args.batch_size)
    print("retain fisher done", flush=True)
    for m in modules.values():
        m.weight.requires_grad_(False)

    def layer_of(name):
        mm = re.search(r"language_model\.layers\.(\d+)\.", name)
        return int(mm.group(1)) if mm else -1

    result = {"adapters": {}}
    for item in args.adapters.split(","):
        name, path = item.split("=", 1)
        ab, scaling = load_adapter_ab(Path(path))
        ab = {norm_name(k): v for k, v in ab.items()}
        common = [k for k in modules if k in ab]
        g_forget = g_retain = 0.0
        per_layer = {}
        for k in common:
            A, B = ab[k]
            dW = (scaling * (B @ A)).float()
            w2 = dW * dW
            vf = float((w2 * f_forget[k]).sum())
            vr = float((w2 * f_retain[k]).sum())
            g_forget += vf
            g_retain += vr
            L = layer_of(k)
            per_layer.setdefault(L, [0.0, 0.0])
            per_layer[L][0] += vf
            per_layer[L][1] += vr
        result["adapters"][name] = {
            "n_modules": len(common),
            "weighted_fisher_forget": g_forget,
            "weighted_fisher_retain": g_retain,
            "ratio_forget_over_retain": g_forget / g_retain if g_retain > 0 else None,
            "per_layer_ratio": {str(L): (v[0] / v[1] if v[1] > 0 else None) for L, v in sorted(per_layer.items())},
        }
        print(f"{name}: ratio={result['adapters'][name]['ratio_forget_over_retain']:.4f}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, v in result["adapters"].items():
        top = sorted(v["per_layer_ratio"].items(), key=lambda kv: -(kv[1] or 0))[:5]
        print(f"{name}: global ratio {v['ratio_forget_over_retain']:.3f}; top layers {top}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
