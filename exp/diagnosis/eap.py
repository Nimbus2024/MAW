#!/usr/bin/env python3
"""EAP-lite: 层级 activation-difference 归因。

对每个层 L, 用一阶近似把行为损失的变化归因到该层的激活差:
  score_L = E[ < dL/dh_L (unlearned), h_L(unlearned) - h_L(base) > ]
(即 EAP 的层粒度版本; score 越大表示该层的激活变化对行为损失上升贡献越大)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..unlearn._paired import build_pairs
from .adapter_geometry import load_adapter_ab
from .patching import apply_delta, batch_to_device, build_batch, get_layers, norm_name, target_modules


def pairs_of(data_split_dir, split, n, seed=42):
    ps = build_pairs(pd.read_parquet(Path(data_split_dir) / split / "train-00000-of-00001.parquet"))
    order = torch.randperm(len(ps), generator=torch.Generator().manual_seed(seed)).tolist()
    return [ps[i] for i in order[:n]]


@torch.no_grad()
def layer_inputs(model, items, modality, layers, device, processor, max_length):
    caps = {}
    handles = []
    for L in layers:
        def make(idx):
            def hook(module, args):
                caps.setdefault(idx, []).append(args[0].detach().clone())
            return hook
        handles.append(get_layers(model)[L].register_forward_pre_hook(make(L)))
    for b in range(0, len(items), 2):
        batch = build_batch(items[b:b + 2], modality, processor, max_length)
        ids, attn, pixel, _ = batch_to_device(batch, modality, device)
        model(input_ids=ids, attention_mask=attn, pixel_values=pixel)
    for h in handles:
        h.remove()
    return caps


def layer_grads(model, items, modality, layers, device, processor, max_length):
    caps, grads = {}, {}
    handles = []
    for L in layers:
        def make(idx):
            def hook(module, args):
                x = args[0]
                x.retain_grad()
                caps.setdefault(idx, []).append(x)
            return hook
        handles.append(get_layers(model)[L].register_forward_pre_hook(make(L)))
    for b in range(0, len(items), 2):
        batch = build_batch(items[b:b + 2], modality, processor, max_length)
        ids, attn, pixel, labels = batch_to_device(batch, modality, device)
        loss = model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels).loss
        loss.backward()
        for L in layers:
            x = caps[L][-1]
            grads.setdefault(L, []).append(x.grad.detach().float().cpu())
            x.grad = None
        model.zero_grad(set_to_none=True)
    for h in handles:
        h.remove()
    return {L: torch.cat(v, 0) for L, v in grads.items()}


def main():
    parser = argparse.ArgumentParser(description="EAP-lite 层级 activation-difference 归因")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", required=True, help="name=path,...")
    parser.add_argument("--data_split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,28,31")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--forget_split", default="forget_5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device)
    model.eval()
    model.config.use_cache = False
    model.requires_grad_(False)
    modules = target_modules(model)
    pairs = pairs_of(args.data_split_dir, args.forget_split, args.n, args.seed)
    items = {"mm": [{"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]} for p in pairs],
             "um": [{"image": None, "question": p["um_q"], "answer": p["um_a"]} for p in pairs]}
    base_h = {m: layer_inputs(model, items[m], m, layers, args.device, processor, args.max_length)
              for m in ("mm", "um")}

    result = {"layers": layers, "adapters": {}}
    for item in args.adapters.split(","):
        name, path = item.split("=", 1)
        ab, scaling = load_adapter_ab(Path(path))
        ab = {norm_name(k): v for k, v in ab.items()}
        apply_delta(modules, ab, scaling, +1.0)
        per = {}
        for m in ("mm", "um"):
            g = layer_grads(model, items[m], m, layers, args.device, processor, args.max_length)
            per[m] = {str(L): float((g[L] * (g[L] - base_h[m][L])).sum() / g[L].shape[0]) for L in layers}
        apply_delta(modules, ab, scaling, -1.0)
        result["adapters"][name] = per
        print(f"{name} done", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("layer   IT_score   PT_score")
    for L in layers:
        it = [result["adapters"][a]["mm"][str(L)] for a in result["adapters"]]
        pt = [result["adapters"][a]["um"][str(L)] for a in result["adapters"]]
        print(f"{L:>5} {sum(it)/len(it):>11.4f} {sum(pt)/len(pt):>10.4f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
