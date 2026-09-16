#!/usr/bin/env python3
"""RepE: 提取模态方向(IT-PT)与知识方向(forget-retain), 逐强度 steer, 测 IT/PT 行为 CE 的变化。

方向取 answer-token mean-pool 的 difference-in-means (单位化); steer = 在该层输入上加 alpha*dir。
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..unlearn._paired import build_pairs, collate_plain
from .adapter_geometry import load_adapter_ab
from .patching import apply_delta, get_layers, norm_name, target_modules, to_device


def samples(data_split_dir, split, n_entities, per_entity, modality, seed=42):
    pairs = build_pairs(pd.read_parquet(Path(data_split_dir) / split / "train-00000-of-00001.parquet"))
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(seed)).tolist()
    by_ent = {}
    for i in order:
        p = pairs[i]
        by_ent.setdefault(p["row"], []).append(p)
    out = []
    for e in list(by_ent)[:n_entities]:
        for p in by_ent[e][:per_entity]:
            out.append({"image": p["image"] if modality == "mm" else None,
                        "question": p["mm_q"] if modality == "mm" else p["um_q"],
                        "answer": p["mm_a"] if modality == "mm" else p["um_a"]})
    return out


def batches(items, modality, processor, max_length, batch_size=4):
    args = types.SimpleNamespace(max_length=max_length)
    out = []
    for b in range(0, len(items), batch_size):
        chunk = items[b:b + batch_size]
        ic = [{"mm": {"image": it["image"], "question": it["question"], "answer": it["answer"]},
               "um": {"question": it["question"], "answer": it["answer"]}} for it in chunk]
        out.append(collate_plain(ic, processor, args))
    return out


@torch.no_grad()
def mean_rep(model, bs, modality, layer, device):
    reps = []
    for batch in bs:
        ids, attn, pixel, labels = to_device(batch, modality, device)
        out = model(input_ids=ids, attention_mask=attn, pixel_values=pixel,
                    output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer].float()
        mask = labels.ne(-100).unsqueeze(-1).float()
        reps.append(((h * mask).sum(1) / mask.sum(1).clamp_min(1)).mean(0).cpu())
    return torch.stack(reps).mean(0)


@torch.no_grad()
def ce_with_dir(model, bs, modality, layer, direction, alpha, device):
    total, n = 0.0, 0
    for batch in bs:
        def hook(module, args):
            x = args[0]
            return (x + alpha * direction.to(x.device, x.dtype),)
        h = get_layers(model)[layer].register_forward_pre_hook(hook)
        try:
            ids, attn, pixel, labels = to_device(batch, modality, device)
            total += float(model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels).loss)
            n += 1
        finally:
            h.remove()
    return total / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="RepE 方向 steer")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", default="")
    parser.add_argument("--data_split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--n_forget", type=int, default=25)
    parser.add_argument("--n_retain", type=int, default=50)
    parser.add_argument("--per_entity", type=int, default=2)
    parser.add_argument("--alphas", default="-2,-1,-0.5,0,0.5,1,2")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device)
    model.eval()
    model.config.use_cache = False
    modules = target_modules(model)

    f_mm = batches(samples(args.data_split_dir, "forget_5", args.n_forget, args.per_entity, "mm", args.seed),
                   "mm", processor, args.max_length)
    f_um = batches(samples(args.data_split_dir, "forget_5", args.n_forget, args.per_entity, "um", args.seed),
                   "um", processor, args.max_length)
    r_mm = batches(samples(args.data_split_dir, "retain_95", args.n_retain, args.per_entity, "mm", args.seed),
                   "mm", processor, args.max_length)
    r_um = batches(samples(args.data_split_dir, "retain_95", args.n_retain, args.per_entity, "um", args.seed),
                   "um", processor, args.max_length)

    mm_it = mean_rep(model, f_mm, "mm", args.layer, args.device)
    mm_pt = mean_rep(model, f_um, "um", args.layer, args.device)
    ret_it = mean_rep(model, r_mm, "mm", args.layer, args.device)
    ret_pt = mean_rep(model, r_um, "um", args.layer, args.device)
    dir_modality = (mm_it - mm_pt)
    dir_modality = dir_modality / dir_modality.norm()
    dir_knowledge = ((mm_it + mm_pt) / 2 - (ret_it + ret_pt) / 2)
    dir_knowledge = dir_knowledge / dir_knowledge.norm()
    print(f"dir modality norm={dir_modality.norm():.3f} dir knowledge norm={dir_knowledge.norm():.3f}", flush=True)

    result = {"layer": args.layer, "alphas": alphas, "models": {}}
    for item in [""] + [a for a in args.adapters.split(",") if a]:
        if item == "":
            name = "base"
        else:
            name, path = item.split("=", 1)
            ab, scaling = load_adapter_ab(Path(path))
            ab = {norm_name(k): v for k, v in ab.items()}
            apply_delta(modules, ab, scaling, +1.0)
        entry = {}
        for dname, direction in (("modality_IT-PT", dir_modality), ("knowledge_forget-retain", dir_knowledge)):
            entry[dname] = {}
            for a in alphas:
                entry[dname][str(a)] = {
                    "it_ce": ce_with_dir(model, f_mm, "mm", args.layer, direction, a, args.device),
                    "pt_ce": ce_with_dir(model, f_um, "um", args.layer, direction, a, args.device),
                }
        result["models"][name] = entry
        if item != "":
            apply_delta(modules, ab, scaling, -1.0)
        print(f"{name} done", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("dir  alpha   IT_CE   PT_CE")
    for dname in result["models"]["base"]:
        for a in alphas:
            e = result["models"]["base"][dname][str(a)]
            print(f"{dname:>22} {a:>5} {e['it_ce']:>8.4f} {e['pt_ce']:>8.4f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
