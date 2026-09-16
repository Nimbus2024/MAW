#!/usr/bin/env python3
"""Activation patching 行为定位: 把 oracle(base) 的层输入激活换进 unlearned 模型,
测 IT / PT 两个行为上"哪些层能恢复正确答案的似然"。

对每个 adapter: 逐层把 base 的该层输入 hidden 替换进 unlearned(原地加 ΔW)前向,
记录答案 CE。基线 = unlearned 未 patch 的 CE; 恢复量 = CE_patch - CE_unlearned (<0 表示恢复)。
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..unlearn._paired import build_pairs, collate_plain
from .adapter_geometry import load_adapter_ab

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MODALITY = {"it": "mm", "pt": "um"}


def norm_name(name: str) -> str:
    return name[name.index("language_model."):] if "language_model." in name else name


def get_layers(model):
    for name, module in model.named_modules():
        if name.endswith("language_model.layers"):
            return module
    raise RuntimeError("language_model.layers not found")


def target_modules(model):
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in TARGET_SUFFIXES:
            out[norm_name(name)] = module
    return out


def probe_batches(data_split_dir, split, n, batch_size, processor, max_length, seed=42):
    pairs = build_pairs(pd.read_parquet(Path(data_split_dir) / split / "train-00000-of-00001.parquet"))
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed)).tolist()
    pairs = [pairs[i] for i in order[:n]]
    args = types.SimpleNamespace(max_length=max_length)
    items = [{"mm": {"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]},
              "um": {"question": p["um_q"], "answer": p["um_a"]}} for p in pairs]
    out = []
    for b in range(0, len(items), batch_size):
        out.append(collate_plain(items[b:b + batch_size], processor, args))
    return out


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


@torch.no_grad()
def ce_loss(model, batch, modality, device):
    ids, attn, pixel, labels = to_device(batch, modality, device)
    return float(model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels).loss)


@torch.no_grad()
def cache_layer_inputs(model, batches, modality, layers, device):
    cache = {}
    handles = []
    for li in layers:
        def make(idx):
            def hook(module, args):
                cache.setdefault(idx, []).append(args[0].detach().clone())
            return hook
        handles.append(get_layers(model)[li].register_forward_pre_hook(make(li)))
    for batch in batches:
        ids, attn, pixel, _ = to_device(batch, modality, device)
        model(input_ids=ids, attention_mask=attn, pixel_values=pixel)
    for h in handles:
        h.remove()
    return cache


@torch.no_grad()
def ce_with_patch(model, batches, modality, layer, ref_list, device):
    total, n = 0.0, 0
    for i, batch in enumerate(batches):
        ref = ref_list[i]
        state = {"done": False}

        def hook(module, args):
            if state["done"]:
                return None
            state["done"] = True
            return (ref.to(args[0].device, args[0].dtype),)

        h = get_layers(model)[layer].register_forward_pre_hook(hook)
        try:
            total += ce_loss(model, batch, modality, device)
            n += 1
        finally:
            h.remove()
    return total / max(n, 1)


def apply_delta(modules, ab, scaling, sign=1.0):
    for name, module in modules.items():
        if name not in ab:
            continue
        A, B = ab[name]
        dW = (scaling * (B @ A)).to(module.weight.device, module.weight.dtype)
        module.weight.data.add_(dW, alpha=sign)


def main():
    parser = argparse.ArgumentParser(description="Activation patching 行为定位")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", required=True, help="name=path,...")
    parser.add_argument("--data_split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
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
    modules = target_modules(model)
    print(f"modules={len(modules)} layers={layers}", flush=True)

    probes = {m: probe_batches(args.data_split_dir, args.forget_split, args.n,
                               args.batch_size, processor, args.max_length, args.seed)
              for m in ("mm", "um")}
    base_ce = {m: sum(ce_loss(model, b, m, args.device) for b in probes[m]) / len(probes[m])
               for m in ("mm", "um")}
    print(f"base CE: {base_ce}", flush=True)
    cache = {m: cache_layer_inputs(model, probes[m], m, layers, args.device) for m in ("mm", "um")}
    print("cached base layer inputs", flush=True)

    result = {"layers": layers, "base_ce": base_ce, "adapters": {}}
    for item in args.adapters.split(","):
        name, path = item.split("=", 1)
        ab, scaling = load_adapter_ab(Path(path))
        ab = {norm_name(k): v for k, v in ab.items()}
        apply_delta(modules, ab, scaling, +1.0)
        unlearned_ce = {m: sum(ce_loss(model, b, m, args.device) for b in probes[m]) / len(probes[m])
                        for m in ("mm", "um")}
        per_mod = {}
        for m in ("mm", "um"):
            patched = [ce_with_patch(model, probes[m], m, L, cache[m][L], args.device) for L in layers]
            per_mod[m] = {
                "unlearned_ce": unlearned_ce[m],
                "patched_ce": {str(L): v for L, v in zip(layers, patched)},
                "recovery": {str(L): unlearned_ce[m] - v for L, v in zip(layers, patched)},
            }
        apply_delta(modules, ab, scaling, -1.0)
        result["adapters"][name] = per_mod
        print(f"{name}: unlearned CE {unlearned_ce}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("layer   IT_unl IT_rec(mean)  PT_unl PT_rec(mean)")
    for L in layers:
        rec_it = [result["adapters"][a]["mm"]["recovery"][str(L)] for a in result["adapters"]]
        rec_pt = [result["adapters"][a]["um"]["recovery"][str(L)] for a in result["adapters"]]
        print(f"{L:>5}  {rec_it[0]:>7.3f} {sum(rec_it)/len(rec_it):>10.3f}  {rec_pt[0]:>7.3f} {sum(rec_pt)/len(rec_pt):>10.3f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
