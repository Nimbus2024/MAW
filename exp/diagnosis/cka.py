#!/usr/bin/env python3
"""CKA + 表示模长 + 逐样本方向变化。

对每层: 取 mean-pool 表示 (layer output, final norm 之前), 计算
  - 线性核 CKA: IT-vs-PT, IT-vs-base, PT-vs-base
  - 表示模长 mean/std (每模态每模型)
  - 逐样本 cos(rep_model, rep_base) 的均值 (方向变化; CKA 对缩放不变, 需 cos 补充)
"""
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..unlearn._paired import build_pairs, collate_plain
from .patching import batch_to_device, build_batch


def cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = (x.T @ y).norm() ** 2
    denom = (x.T @ x).norm() * (y.T @ y).norm()
    return float(hsic / denom) if denom > 0 else float("nan")


def probe_pairs(data_split_dir, split, n, seed=42):
    ps = build_pairs(pd.read_parquet(Path(data_split_dir) / split / "train-00000-of-00001.parquet"))
    order = torch.randperm(len(ps), generator=torch.Generator().manual_seed(seed)).tolist()
    return [ps[i] for i in order[:n]]


@torch.no_grad()
def reps(model, pairs, modality, layers, device, processor, max_length, batch_size=4):
    out = {L: [] for L in layers}
    for b in range(0, len(pairs), batch_size):
        chunk = pairs[b:b + batch_size]
        items = [{"image": p["image"] if modality == "mm" else None,
                  "question": p["mm_q"] if modality == "mm" else p["um_q"],
                  "answer": p["mm_a"] if modality == "mm" else p["um_a"]} for p in chunk]
        batch = build_batch(items, modality, processor, max_length)
        ids, attn, pixel, labels = batch_to_device(batch, modality, device)
        o = model(input_ids=ids, attention_mask=attn, pixel_values=pixel,
                  output_hidden_states=True, use_cache=False)
        mask = attn.unsqueeze(-1).float()
        for L in layers:
            h = o.hidden_states[L].float()
            out[L].append(((h * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu())
    return {L: torch.cat(v, 0) for L, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description="CKA + 模长 + 方向变化")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", default="")
    parser.add_argument("--data_split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,28,31")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--forget_split", default="forget_5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    base = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device).eval()
    base.config.use_cache = False
    pairs = probe_pairs(args.data_split_dir, args.forget_split, args.n, args.seed)

    reps_all = {"base": {m: reps(base, pairs, m, layers, args.device, processor, args.max_length, args.batch_size)
                         for m in ("mm", "um")}}
    print("base reps done", flush=True)

    adapter_list = [it.split("=", 1) for it in args.adapters.split(",") if it]
    if adapter_list:
        peft = PeftModel.from_pretrained(base, adapter_list[0][1], adapter_name=adapter_list[0][0])
        peft.eval()
        for name, path in adapter_list:
            if name != adapter_list[0][0]:
                peft.load_adapter(path, adapter_name=name)
            peft.set_adapter(name)
            reps_all[name] = {m: reps(peft, pairs, m, layers, args.device, processor, args.max_length, args.batch_size)
                              for m in ("mm", "um")}
            print(f"{name} reps done", flush=True)

    result = {"layers": layers, "n": len(pairs), "models": {}}
    for name, per_mod in reps_all.items():
        entry = {}
        for L in layers:
            it, pt = per_mod["mm"][L], per_mod["um"][L]
            e = {"cka_it_pt": cka(it, pt),
                 "norm_it": float(it.norm(dim=1).mean()),
                 "norm_pt": float(pt.norm(dim=1).mean())}
            if name != "base":
                b_it, b_pt = reps_all["base"]["mm"][L], reps_all["base"]["um"][L]
                e["cka_it_base"] = cka(it, b_it)
                e["cka_pt_base"] = cka(pt, b_pt)
                e["cos_it_base"] = float(torch.nn.functional.cosine_similarity(it, b_it, dim=1).mean())
                e["cos_pt_base"] = float(torch.nn.functional.cosine_similarity(pt, b_pt, dim=1).mean())
                e["norm_ratio_it"] = e["norm_it"] / float(b_it.norm(dim=1).mean())
                e["norm_ratio_pt"] = e["norm_pt"] / float(b_pt.norm(dim=1).mean())
            entry[str(L)] = e
        result["models"][name] = entry

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("layer | base CKA(IT,PT) norm(IT,PT) | joint CKA(IT,PT) vsbase(IT,PT) cos(IT,PT) normratio(IT,PT)")
    for L in layers:
        b = result["models"]["base"][str(L)]
        j = result["models"].get("joint", {}).get(str(L), {})
        print(f"{L:>5} | {b['cka_it_pt']:.3f} n=({b['norm_it']:.0f},{b['norm_pt']:.0f}) | "
              f"{j.get('cka_it_pt', float('nan')):.3f} "
              f"({j.get('cka_it_base', float('nan')):.3f},{j.get('cka_pt_base', float('nan')):.3f}) "
              f"cos({j.get('cos_it_base', float('nan')):.3f},{j.get('cos_pt_base', float('nan')):.3f}) "
              f"nr({j.get('norm_ratio_it', float('nan')):.3f},{j.get('norm_ratio_pt', float('nan')):.3f})")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
