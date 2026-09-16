#!/usr/bin/env python3
"""Activation patching 行为定位。

逐层把 oracle(base) 的层输入激活换进 unlearned 模型, 测 IT / PT 两个行为上的恢复。
指标二选一:
  --metric ce  : 答案 token 的 CE (recovery = unlearned_CE - patched_CE, 越大越恢复)
  --metric gen : 贪心生成后 fuzzy 判分命中率 (recovery = patched_acc - unlearned_acc)
"""
from __future__ import annotations

import argparse
import json
import re
import types
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..eval.eval_vllm import _judge_answer
from ..unlearn._paired import build_pairs, collate_plain
from .adapter_geometry import load_adapter_ab

TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


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
    items, answers = [], {"mm": [], "um": []}
    for p in pairs:
        items.append({"mm": {"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]},
                      "um": {"question": p["um_q"], "answer": p["um_a"]}})
        answers["mm"].append(p["mm_a"])
        answers["um"].append(p["um_a"])
    out = []
    for b in range(0, len(items), batch_size):
        out.append({"batch": collate_plain(items[b:b + batch_size], processor, args),
                    "answers": {m: answers[m][b:b + batch_size] for m in ("mm", "um")}})
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
def cache_layer_inputs(model, probes, modality, layers, device):
    cache = {}
    handles = []
    for li in layers:
        def make(idx):
            def hook(module, args):
                cache.setdefault(idx, []).append(args[0].detach().clone())
            return hook
        handles.append(get_layers(model)[li].register_forward_pre_hook(make(li)))
    for item in probes:
        ids, attn, pixel, _ = to_device(item["batch"], modality, device)
        model(input_ids=ids, attention_mask=attn, pixel_values=pixel)
    for h in handles:
        h.remove()
    return cache


def _patched_call(model, item, modality, layer, ref, device, metric, max_new_tokens, processor):
    state = {"done": False}

    def hook(module, args):
        if state["done"]:
            return None
        state["done"] = True
        return (ref.to(args[0].device, args[0].dtype),)

    h = get_layers(model)[layer].register_forward_pre_hook(hook)
    try:
        ids, attn, pixel, labels = to_device(item["batch"], modality, device)
        if metric == "ce":
            return float(model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels).loss)
        out = model.generate(input_ids=ids, attention_mask=attn, pixel_values=pixel,
                             max_new_tokens=max_new_tokens, do_sample=False)
        gen = processor.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
        return sum(_judge_answer(g, a) for g, a in zip(gen, item["answers"][modality])) / len(gen)
    finally:
        h.remove()


def apply_delta(modules, ab, scaling, sign=1.0):
    for name, module in modules.items():
        if name not in ab:
            continue
        A, B = ab[name]
        dW = (scaling * (B @ A)).to(module.weight.device, module.weight.dtype)
        module.weight.data.add_(dW, alpha=sign)


@torch.no_grad()
def baseline_metric(model, probes, modality, device, metric, max_new_tokens, processor):
    if metric == "ce":
        return sum(ce_loss(model, it["batch"], modality, device) for it in probes) / len(probes)
    total, n = 0.0, 0
    for item in probes:
        ids, attn, pixel, _ = to_device(item["batch"], modality, device)
        out = model.generate(input_ids=ids, attention_mask=attn, pixel_values=pixel,
                             max_new_tokens=max_new_tokens, do_sample=False)
        gen = processor.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
        total += sum(_judge_answer(g, a) for g, a in zip(gen, item["answers"][modality])) / len(gen)
        n += 1
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Activation patching 行为定位")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", required=True, help="name=path,...")
    parser.add_argument("--data_split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", choices=("ce", "gen"), default="ce")
    parser.add_argument("--layers", default="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=24)
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
    print(f"modules={len(modules)} metric={args.metric} layers={layers}", flush=True)

    probes = {m: probe_batches(args.data_split_dir, args.forget_split, args.n,
                               args.batch_size, processor, args.max_length, args.seed)
              for m in ("mm", "um")}
    base_m = {m: baseline_metric(model, probes[m], m, args.device, args.metric,
                                 args.max_new_tokens, processor) for m in ("mm", "um")}
    print(f"base {args.metric}: {base_m}", flush=True)
    cache = {m: cache_layer_inputs(model, probes[m], m, layers, args.device) for m in ("mm", "um")}
    print("cached base layer inputs", flush=True)

    result = {"metric": args.metric, "layers": layers, "base": base_m, "adapters": {}}
    for item in args.adapters.split(","):
        name, path = item.split("=", 1)
        ab, scaling = load_adapter_ab(Path(path))
        ab = {norm_name(k): v for k, v in ab.items()}
        apply_delta(modules, ab, scaling, +1.0)
        unl = {m: baseline_metric(model, probes[m], m, args.device, args.metric,
                                  args.max_new_tokens, processor) for m in ("mm", "um")}
        per_mod = {}
        for m in ("mm", "um"):
            patched = []
            for L in layers:
                vals = [_patched_call(model, it, m, L, cache[m][L][i], args.device, args.metric,
                                      args.max_new_tokens, processor)
                        for i, it in enumerate(probes[m])]
                patched.append(sum(vals) / len(vals))
            per_mod[m] = {"unlearned": unl[m],
                          "patched": {str(L): v for L, v in zip(layers, patched)},
                          "recovery": {str(L): (unl[m] - v if args.metric == "ce" else v - unl[m])
                                       for L, v in zip(layers, patched)}}
        apply_delta(modules, ab, scaling, -1.0)
        result["adapters"][name] = per_mod
        print(f"{name}: unlearned={unl}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("layer   IT_rec(mean)  PT_rec(mean)")
    for L in layers:
        it = [result["adapters"][a]["mm"]["recovery"][str(L)] for a in result["adapters"]]
        pt = [result["adapters"][a]["um"]["recovery"][str(L)] for a in result["adapters"]]
        print(f"{L:>5} {sum(it)/len(it):>12.3f} {sum(pt)/len(pt):>12.3f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
