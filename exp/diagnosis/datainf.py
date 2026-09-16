#!/usr/bin/env python3
"""DataInf-lite: 逐训练样本对 unlearning 更新的影响归因 (LoRA 参数空间)。

对最终 adapter (PeftModel) 的每个 forget 样本 i 计算 LoRA 参数梯度 g_i,
影响 I_i = <g_i, Δθ> (Δθ = adapter 参数本身, 因 θ0=0)。
报告按 |I_i| 排序的样本、以及 mm/um 两个模态的均值。
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

from ..unlearn._paired import build_pairs
from .patching import batch_to_device, build_batch


def pairs_of(data_split_dir, split, n, seed=42):
    ps = build_pairs(pd.read_parquet(Path(data_split_dir) / split / "train-00000-of-00001.parquet"))
    order = torch.randperm(len(ps), generator=torch.Generator().manual_seed(seed)).tolist()
    return [ps[i] for i in order[:n]]


def main():
    parser = argparse.ArgumentParser(description="DataInf-lite 样本影响归因")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data_split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_samples", type=int, default=120)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--forget_split", default="forget_5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    base = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True).to(args.device)
    model.eval()
    model.config.use_cache = False
    lora_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"lora params: {len(lora_params)}", flush=True)

    pairs = pairs_of(args.data_split_dir, args.forget_split, args.n_samples, args.seed)
    records = []
    for i, p in enumerate(pairs):
        for modality in ("mm", "um"):
            item = {"image": p["image"] if modality == "mm" else None,
                    "question": p["mm_q"] if modality == "mm" else p["um_q"],
                    "answer": p["mm_a"] if modality == "mm" else p["um_a"]}
            batch = build_batch([item], modality, processor, args.max_length)
            ids, attn, pixel, labels = batch_to_device(batch, modality, args.device)
            loss = model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels).loss
            loss.backward()
            infl = 0.0
            for _, param in lora_params:
                if param.grad is not None:
                    infl += float((param.grad.detach().float() * param.detach().float()).sum())
            model.zero_grad(set_to_none=True)
            records.append({"entity": int(p["row"]), "modality": modality, "influence": infl,
                            "loss": float(loss)})
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(pairs)}", flush=True)

    by_mod = {}
    for m in ("mm", "um"):
        vals = [r["influence"] for r in records if r["modality"] == m]
        by_mod[m] = {"mean": sum(vals) / len(vals), "n": len(vals)}
    top = sorted(records, key=lambda r: -abs(r["influence"]))[:20]
    result = {"n_pairs": len(pairs), "by_modality": by_mod,
              "top_abs": top, "records": records}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("by modality:", {m: round(v["mean"], 5) for m, v in by_mod.items()})
    print("top5:", [(r["modality"], r["entity"], round(r["influence"], 5)) for r in top[:5]])
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
