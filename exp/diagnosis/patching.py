#!/usr/bin/env python3
"""Activation patching 行为定位。

逐层把 oracle(base) 的层输入激活换进 unlearned 模型, 测 IT / PT 两个行为上的恢复。
  --metric ce  : 答案 token CE (recovery = unlearned_CE - patched_CE, 越大越恢复)
  --metric gen : 纯提示贪心生成 + fuzzy 判分命中率 (recovery = patched_acc - unlearned_acc)
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..eval.eval_vllm import _judge_answer
from ..unlearn._paired import build_pairs, collate_plain
from ..unlearn.unlearn_dataset import (
    train_collate_fn_llava_multimodal,
    train_collate_fn_llava_unimodal,
)
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


def load_pairs(data_split_dir, split, n, seed=42):
    pairs = build_pairs(pd.read_parquet(Path(data_split_dir) / split / "train-00000-of-00001.parquet"))
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed)).tolist()
    return [pairs[i] for i in order[:n]]


def ce_probes(pairs, processor, max_length, batch_size):
    args = types.SimpleNamespace(max_length=max_length)
    items = [{"mm": {"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]},
              "um": {"question": p["um_q"], "answer": p["um_a"]}} for p in pairs]
    out = []
    for b in range(0, len(items), batch_size):
        out.append({"batch": collate_plain(items[b:b + batch_size], processor, args),
                    "answers": {m: [it[m]["answer"] for it in items[b:b + batch_size]] for m in ("mm", "um")}})
    return out


def gen_items(pairs, modality):
    out = []
    for p in pairs:
        if modality == "mm":
            out.append({"prompt": f"USER: <image>\n{p['mm_q']}\nASSISTANT:", "image": p["image"], "answer": p["mm_a"]})
        else:
            out.append({"prompt": f"USER: {p['um_q']}\nASSISTANT:", "image": None, "answer": p["um_a"]})
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


def build_batch(chunk, modality, processor, max_length):
    args = types.SimpleNamespace(max_length=max_length)
    if modality == "mm":
        items = [{"image": it["image"], "question": it["question"], "answer": it["answer"]} for it in chunk]
        return train_collate_fn_llava_multimodal(items, processor, args)
    items = [{"question": it["question"], "answer": it["answer"]} for it in chunk]
    return train_collate_fn_llava_unimodal(items, processor, args)


def batch_to_device(batch, modality, device):
    ids, attn, pixel, labels = batch
    ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
    if modality == "mm" and pixel is not None:
        pixel = pixel.to(device)
    else:
        pixel = None
    return ids, attn, pixel, labels


def tokenize(item, modality, processor, device):
    images = [item["image"]] if modality == "mm" else None
    inputs = processor(text=[item["prompt"]], images=images, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


@torch.no_grad()
def ce_loss(model, batch, modality, device):
    ids, attn, pixel, labels = to_device(batch, modality, device)
    logits = model(input_ids=ids, attention_mask=attn, pixel_values=pixel).logits
    lg = logits[:, :-1].float()
    sl = labels[:, 1:]
    ls = torch.nn.functional.cross_entropy(
        lg.reshape(-1, lg.size(-1)), sl.reshape(-1), reduction="none").view(sl.shape)
    m = sl != -100
    per_sample = (ls * m).sum(1) / m.sum(1).clamp_min(1)
    return float(per_sample.mean())


@torch.no_grad()
def ce_baseline(model, probes, modality, device):
    return sum(ce_loss(model, it["batch"], modality, device) for it in probes) / len(probes)


@torch.no_grad()
def gen_baseline(model, items, modality, device, max_new_tokens, processor):
    total = 0.0
    for it in items:
        inputs = tokenize(it, modality, processor, device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        total += _judge_answer(gen[0], it["answer"])
    return total / len(items)


@torch.no_grad()
def cache_ce_inputs(model, probes, modality, layers, device):
    cache = {}
    handles = [get_layers(model)[li].register_forward_pre_hook(
        (lambda idx: (lambda m, a: cache.setdefault(idx, []).append(a[0].detach().clone())))(li))
        for li in layers]
    for it in probes:
        ids, attn, pixel, _ = to_device(it["batch"], modality, device)
        model(input_ids=ids, attention_mask=attn, pixel_values=pixel)
    for h in handles:
        h.remove()
    return cache


@torch.no_grad()
def cache_gen_inputs(model, items, modality, layers, device, processor):
    cache = {}
    handles = [get_layers(model)[li].register_forward_pre_hook(
        (lambda idx: (lambda m, a: cache.setdefault(idx, []).append(a[0].detach().clone())))(li))
        for li in layers]
    for it in items:
        model(**tokenize(it, modality, processor, device))
    for h in handles:
        h.remove()
    return cache


def _hook_ref(model, layer, ref):
    state = {"done": False}

    def hook(module, args):
        if state["done"]:
            return None
        state["done"] = True
        return (ref.to(args[0].device, args[0].dtype),)
    return get_layers(model)[layer].register_forward_pre_hook(hook)


@torch.no_grad()
def ce_patched(model, batch, modality, layer, ref, device):
    h = _hook_ref(model, layer, ref)
    try:
        return ce_loss(model, batch, modality, device)
    finally:
        h.remove()


@torch.no_grad()
def gen_patched(model, item, modality, layer, ref, device, max_new_tokens, processor):
    inputs = tokenize(item, modality, processor, device)
    h = _hook_ref(model, layer, ref)
    try:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _judge_answer(gen[0], item["answer"])
    finally:
        h.remove()


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
    parser.add_argument("--metric", choices=("ce", "gen"), default="ce")
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,28,31")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
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
    pairs = load_pairs(args.data_split_dir, args.forget_split, args.n, args.seed)

    # base 参照的层输入缓存（按模态）
    if args.metric == "ce":
        probes_all = {m: ce_probes(pairs, processor, args.max_length, args.batch_size) for m in ("mm", "um")}
        cache = {m: cache_ce_inputs(model, probes_all[m], m, layers, args.device) for m in ("mm", "um")}
        base_metric = {m: ce_baseline(model, probes_all[m], m, args.device) for m in ("mm", "um")}
    else:
        items_all = {m: gen_items(pairs, m) for m in ("mm", "um")}
        cache = {m: cache_gen_inputs(model, items_all[m], m, layers, args.device, processor) for m in ("mm", "um")}
        base_metric = {m: gen_baseline(model, items_all[m], m, args.device, args.max_new_tokens, processor)
                       for m in ("mm", "um")}
    print(f"base {args.metric}: {base_metric}", flush=True)

    result = {"metric": args.metric, "layers": layers, "base": base_metric, "adapters": {}}
    for item in args.adapters.split(","):
        name, path = item.split("=", 1)
        ab, scaling = load_adapter_ab(Path(path))
        ab = {norm_name(k): v for k, v in ab.items()}
        apply_delta(modules, ab, scaling, +1.0)
        per_mod = {}
        for m in ("mm", "um"):
            if args.metric == "ce":
                unl = ce_baseline(model, probes_all[m], m, args.device)
                patched = [sum(ce_patched(model, it["batch"], m, L, cache[m][L][i], args.device)
                               for i, it in enumerate(probes_all[m])) / len(probes_all[m]) for L in layers]
                rec = {str(L): unl - v for L, v in zip(layers, patched)}
            else:
                unl = gen_baseline(model, items_all[m], m, args.device, args.max_new_tokens, processor)
                patched = [sum(gen_patched(model, it, m, L, cache[m][L][i], args.device,
                                           args.max_new_tokens, processor)
                               for i, it in enumerate(items_all[m])) / len(items_all[m]) for L in layers]
                rec = {str(L): v - unl for L, v in zip(layers, patched)}
            per_mod[m] = {"unlearned": unl, "patched": {str(L): v for L, v in zip(layers, patched)},
                          "recovery": rec}
        apply_delta(modules, ab, scaling, -1.0)
        result["adapters"][name] = per_mod
        print(f"{name}: unlearned={ {m: round(per_mod[m]['unlearned'],4) for m in per_mod} }", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("layer   IT_rec(mean)  PT_rec(mean)")
    for L in layers:
        it = [result["adapters"][a]["mm"]["recovery"][str(L)] for a in result["adapters"]]
        pt = [result["adapters"][a]["um"]["recovery"][str(L)] for a in result["adapters"]]
        print(f"{L:>5} {sum(it)/len(it):>12.4f} {sum(pt)/len(pt):>12.4f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
