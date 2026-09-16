#!/usr/bin/env python3
"""Activation patching 的跨模型有效性对照。

对 joint adapter, 逐层测:
  base_ce   : base 模型未 patch
  unl_ce    : unlearned 未 patch
  fwd_ce    : base h_L -> unlearned  (正向 patching)
  rev_ce    : unlearned h_L -> base  (反向 patching; 2x2 的另一格)
  rand_ce   : 打乱的 base h_L -> unlearned (随机激活对照)
  align_ce  : Procrustes(W) 对齐后的 base h_L -> unlearned (消坐标差异)
  self_ce   : unlearned h_L -> unlearned (空对照, 应 = unl_ce)
另报 fwd 与 rev 的对称性, 以判断两表示空间是否兼容。
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


def pairs_of(data_split_dir, split, n, offset, seed=42):
    ps = build_pairs(pd.read_parquet(Path(data_split_dir) / split / "train-00000-of-00001.parquet"))
    order = torch.randperm(len(ps), generator=torch.Generator().manual_seed(seed)).tolist()
    ps = [ps[i] for i in order]
    return ps[offset:offset + n]


def make_items(pairs, modality):
    return [{"image": p["image"] if modality == "mm" else None,
             "question": p["mm_q"] if modality == "mm" else p["um_q"],
             "answer": p["mm_a"] if modality == "mm" else p["um_a"]} for p in pairs]


@torch.no_grad()
def cache_h(model, items, modality, layers, device, processor, max_length):
    caps = {L: [] for L in layers}
    handles = []
    for L in layers:
        def make(idx):
            def hook(module, args):
                caps[idx].append(args[0].detach().clone())
            return hook
        handles.append(get_layers(model)[L].register_forward_pre_hook(make(L)))
    for it in items:
        batch = build_batch([it], modality, processor, max_length)
        ids, attn, pixel, _ = batch_to_device(batch, modality, device)
        model(input_ids=ids, attention_mask=attn, pixel_values=pixel)
    for h in handles:
        h.remove()
    return caps


@torch.no_grad()
def ce_with(model, items, modality, device, processor, max_length, layer=None, ref=None):
    total = 0.0
    for i, it in enumerate(items):
        batch = build_batch([it], modality, processor, max_length)
        ids, attn, pixel, labels = batch_to_device(batch, modality, device)
        if layer is None:
            total += float(model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels).loss)
        else:
            def hook(module, args):
                return (ref[i].to(args[0].device, args[0].dtype),)
            h = get_layers(model)[layer].register_forward_pre_hook(hook)
            try:
                total += float(model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels).loss)
            finally:
                h.remove()
    return total / len(items)


def procrustes(X, Y):
    U, _, Vt = torch.linalg.svd(X.T @ Y, full_matrices=False)
    return U @ Vt


def main():
    parser = argparse.ArgumentParser(description="patching 跨模型对照")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data_split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="8,16,24,31")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--n_fit", type=int, default=16)
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
    ab, scaling = load_adapter_ab(Path(args.adapter))
    ab = {norm_name(k): v for k, v in ab.items()}

    probes = {m: make_items(pairs_of(args.data_split_dir, args.forget_split, args.n, 0, args.seed), m)
              for m in ("mm", "um")}
    fit = {m: make_items(pairs_of(args.data_split_dir, args.forget_split, args.n_fit, args.n, args.seed), m)
           for m in ("mm", "um")}

    result = {"layers": layers, "n": args.n, "n_fit": args.n_fit, "adapters": {}}
    for m in ("mm", "um"):
        base_h = cache_h(model, probes[m], m, layers, args.device, processor, args.max_length)
        apply_delta(modules, ab, scaling, +1.0)
        unl_h = cache_h(model, probes[m], m, layers, args.device, processor, args.max_length)
        unl_ce = ce_with(model, probes[m], m, args.device, processor, args.max_length)
        apply_delta(modules, ab, scaling, -1.0)
        base_ce = ce_with(model, probes[m], m, args.device, processor, args.max_length)

        fit_base = cache_h(model, fit[m], m, layers, args.device, processor, args.max_length)
        apply_delta(modules, ab, scaling, +1.0)
        fit_unl = cache_h(model, fit[m], m, layers, args.device, processor, args.max_length)
        apply_delta(modules, ab, scaling, -1.0)

        per = {}
        for L in layers:
            X = torch.cat([h.reshape(-1, h.shape[-1]) for h in fit_base[L]], 0).float()
            Y = torch.cat([h.reshape(-1, h.shape[-1]) for h in fit_unl[L]], 0).float()
            W = procrustes(X, Y)
            zero_ref = [torch.zeros_like(h) for h in base_h[L]]
            shuf_ref = [h[:, torch.randperm(h.shape[1]), :] for h in base_h[L]]
            apply_delta(modules, ab, scaling, +1.0)
            fwd = ce_with(model, probes[m], m, args.device, processor, args.max_length, L, base_h[L])
            rnd = ce_with(model, probes[m], m, args.device, processor, args.max_length, L, zero_ref)
            rnd_unl = ce_with(model, probes[m], m, args.device, processor, args.max_length, L, shuf_ref)
            aligned = [W @ h.reshape(-1, h.shape[-1]).float() for h in base_h[L]]
            aligned = [a.reshape(base_h[L][i].shape).to(base_h[L][i].dtype) for i, a in enumerate(aligned)]
            aln = ce_with(model, probes[m], m, args.device, processor, args.max_length, L, aligned)
            selfc = ce_with(model, probes[m], m, args.device, processor, args.max_length, L, unl_h[L])
            apply_delta(modules, ab, scaling, -1.0)
            rev = ce_with(model, probes[m], m, args.device, processor, args.max_length, L, unl_h[L])
            per[str(L)] = {"base_ce": base_ce, "unl_ce": unl_ce, "fwd_ce": fwd, "rev_ce": rev,
                           "zero_ce": rnd, "shuffle_ce": rnd_unl, "align_ce": aln, "self_ce": selfc}
            print(f"{m} L{L}: base={base_ce:.4f} unl={unl_ce:.4f} fwd={fwd:.4f} rev={rev:.4f} "
                  f"zero={rnd:.4f} shuffle={rnd_unl:.4f} align={aln:.4f} self={selfc:.4f}", flush=True)
        result["adapters"][m] = per

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
