#!/usr/bin/env python3
"""逐层线性探针: 用 answer-token mean-pool 的 hidden state, 逐层训练 LogisticRegression,
测 (a) IT vs PT 模态可分性, (b) forget vs retain 实体可分性 (知识存在信号)。

对 base 与各 adapter 分别做; 实体级 70/30 split 防泄漏。
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..unlearn._paired import build_pairs, collate_plain
from .adapter_geometry import load_adapter_ab
from .patching import apply_delta, norm_name, target_modules


def samples(data_split_dir, split, n_entities, per_entity, modality, processor, max_length, seed=42):
    pairs = build_pairs(pd.read_parquet(Path(data_split_dir) / split / "train-00000-of-00001.parquet"))
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pairs), generator=rng).tolist()
    pairs = [pairs[i] for i in order]
    by_ent = {}
    for p in pairs:
        by_ent.setdefault(p["row"], []).append(p)
    ents = list(by_ent)[:n_entities]
    out = []
    for e in ents:
        for p in by_ent[e][:per_entity]:
            out.append({"entity": e, "image": p["image"] if modality == "mm" else None,
                        "question": p["mm_q"] if modality == "mm" else p["um_q"],
                        "answer": p["mm_a"] if modality == "mm" else p["um_a"]})
    return out


@torch.no_grad()
def hidden_pool(model, items, processor, layers, device, max_length, batch_size=4):
    reps = {L: [] for L in layers}
    for b in range(0, len(items), batch_size):
        chunk = items[b:b + batch_size]
        args = types.SimpleNamespace(max_length=max_length)
        items_c = [{"mm": {"image": it["image"], "question": it["question"], "answer": it["answer"]},
                    "um": {"question": it["question"], "answer": it["answer"]}} for it in chunk]
        batch = collate_plain(items_c, processor, args)
        mm, um = batch["mm"], batch["um"]
        if chunk[0]["image"] is not None:
            ids, attn, pixel, labels = mm
        else:
            ids, attn, _, labels = um
            pixel = None
        ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
        if pixel is not None:
            pixel = pixel.to(device)
        out = model(input_ids=ids, attention_mask=attn, pixel_values=pixel,
                    output_hidden_states=True, use_cache=False)
        mask = labels.ne(-100).unsqueeze(-1).float()
        for L in layers:
            h = out.hidden_states[L].float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
            reps[L].append(pooled.cpu())
    return {L: torch.cat(v, 0).numpy() for L, v in reps.items()}


def probe_accuracy(X, y, entities, seed=42):
    uniq = sorted(set(entities))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    test_ents = set(uniq[:max(1, len(uniq) // 3)])
    tr = [i for i, e in enumerate(entities) if e not in test_ents]
    te = [i for i, e in enumerate(entities) if e in test_ents]
    if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
        return None
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X[tr], y[tr])
    return float(clf.score(X[te], y[te]))


def main():
    parser = argparse.ArgumentParser(description="逐层线性探针")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", default="", help="name=path,...")
    parser.add_argument("--data_split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,28,31")
    parser.add_argument("--n_forget", type=int, default=25)
    parser.add_argument("--n_retain", type=int, default=50)
    parser.add_argument("--per_entity", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
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

    f_mm = samples(args.data_split_dir, "forget_5", args.n_forget, args.per_entity, "mm", processor, args.max_length, args.seed)
    f_um = samples(args.data_split_dir, "forget_5", args.n_forget, args.per_entity, "um", processor, args.max_length, args.seed)
    r_mm = samples(args.data_split_dir, "retain_95", args.n_retain, args.per_entity, "mm", processor, args.max_length, args.seed)
    r_um = samples(args.data_split_dir, "retain_95", args.n_retain, args.per_entity, "um", processor, args.max_length, args.seed)
    items = {"mm": f_mm + r_mm, "um": f_um + r_um}
    labels = {"mm": [1] * len(f_mm) + [0] * len(r_mm), "um": [1] * len(f_um) + [0] * len(r_um)}
    ents = {"mm": [s["entity"] for s in items["mm"]], "um": [s["entity"] for s in items["um"]]}

    result = {"layers": layers, "models": {}}
    for item in [""] + [a for a in args.adapters.split(",") if a]:
        if item == "":
            name = "base"
        else:
            name, path = item.split("=", 1)
            ab, scaling = load_adapter_ab(Path(path))
            ab = {norm_name(k): v for k, v in ab.items()}
            apply_delta(modules, ab, scaling, +1.0)
        reps = {m: hidden_pool(model, items[m], processor, layers, args.device, args.max_length)
                for m in ("mm", "um")}
        per = {}
        for L in layers:
            per[str(L)] = {}
            for m in ("mm", "um"):
                X = reps[m][L]
                y = np.array(labels[m])
                per[str(L)][m] = probe_accuracy(X, y, ents[m], args.seed)
        result["models"][name] = per
        if item != "":
            apply_delta(modules, ab, scaling, -1.0)
        print(f"{name} done", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("layer  base_mm base_um  joint_mm joint_um")
    for L in layers:
        jm = result["models"].get("joint", {}).get(str(L), {})
        print(f"{L:>5} {result['models']['base'][str(L)]['mm']:>8} {result['models']['base'][str(L)]['um']:>7} "
              f"{jm.get('mm')} {jm.get('um')}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
