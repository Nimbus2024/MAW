#!/usr/bin/env python3
"""Logit lens: 逐层把 hidden state 经 final norm + lm_head 解码, 测答案 token 的 CE,
看"答案在哪一层形成", 以及 unlearn 对各层解码的影响 (IT vs PT)。"""
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
from .patching import apply_delta, norm_name, target_modules, to_device


def probe_batches(data_split_dir, split, n, batch_size, processor, max_length, seed=42):
    pairs = build_pairs(pd.read_parquet(Path(data_split_dir) / split / "train-00000-of-00001.parquet"))
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed)).tolist()
    pairs = [pairs[i] for i in order[:n]]
    args = types.SimpleNamespace(max_length=max_length)
    items = [{"mm": {"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]},
              "um": {"question": p["um_q"], "answer": p["um_a"]}} for p in pairs]
    return [collate_plain(items[b:b + batch_size], processor, args)
            for b in range(0, len(items), batch_size)]


def final_norm_and_head(model):
    norm = None
    for name, module in model.named_modules():
        if name.endswith("language_model.norm"):
            norm = module
    head = model.get_output_embeddings()
    return norm, head


@torch.no_grad()
def per_layer_ce(model, batches, modality, layers, device):
    norm, head = final_norm_and_head(model)
    totals = {L: 0.0 for L in layers}
    n = 0
    for batch in batches:
        ids, attn, pixel, labels = to_device(batch, modality, device)
        out = model(input_ids=ids, attention_mask=attn, pixel_values=pixel,
                    output_hidden_states=True, use_cache=False)
        shift_labels = labels[:, 1:]
        for L in layers:
            h = out.hidden_states[L]
            logits = head(norm(h)).float()
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   shift_labels.reshape(-1), ignore_index=-100)
            totals[L] += float(loss)
        n += 1
    return {L: totals[L] / max(n, 1) for L in layers}


def main():
    parser = argparse.ArgumentParser(description="Logit lens 逐层答案 CE")
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

    probes = {m: probe_batches(args.data_split_dir, args.forget_split, args.n,
                               args.batch_size, processor, args.max_length, args.seed)
              for m in ("mm", "um")}
    result = {"layers": layers, "models": {}}
    result["models"]["base"] = {m: {str(L): v for L, v in per_layer_ce(model, probes[m], m, layers, args.device).items()}
                                for m in ("mm", "um")}
    print("base done", flush=True)

    for item in args.adapters.split(","):
        name, path = item.split("=", 1)
        ab, scaling = load_adapter_ab(Path(path))
        ab = {norm_name(k): v for k, v in ab.items()}
        apply_delta(modules, ab, scaling, +1.0)
        result["models"][name] = {m: {str(L): v for L, v in per_layer_ce(model, probes[m], m, layers, args.device).items()}
                                  for m in ("mm", "um")}
        apply_delta(modules, ab, scaling, -1.0)
        print(f"{name} done", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("layer  base_IT base_PT  joint_IT joint_PT")
    for L in layers:
        print(f"{L:>5} {result['models']['base']['mm'][str(L)]:>8.3f} {result['models']['base']['um'][str(L)]:>8.3f} "
              f"{result['models'].get('joint', {}).get('mm', {}).get(str(L), float('nan')):>9.3f} "
              f"{result['models'].get('joint', {}).get('um', {}).get(str(L), float('nan')):>9.3f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
