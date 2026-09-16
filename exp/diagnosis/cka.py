#!/usr/bin/env python3
"""CKA 表示对比: base vs 各 unlearned adapter, 以及 IT vs PT。

对同一批 probe 样本(IT 与 PT 各 N 个), 取指定层 hidden state 的 mean-pool 表示,
计算线性核 CKA: base-vs-unlearned(每模态)、IT-vs-PT(每模型)。
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


def cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    hsic = (x.T @ y).norm() ** 2
    denom = (x.T @ x).norm() * (y.T @ y).norm()
    return float(hsic / denom) if denom > 0 else float("nan")


def probe_pairs(data_split_dir: str, split: str, n: int, processor, max_length: int, seed: int = 42):
    path = Path(data_split_dir) / split / "train-00000-of-00001.parquet"
    pairs = build_pairs(pd.read_parquet(path))
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pairs), generator=rng).tolist()
    pairs = [pairs[i] for i in order[:n]]
    args = types.SimpleNamespace(max_length=max_length)
    mm = [{"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]} for p in pairs]
    um = [{"question": p["um_q"], "answer": p["um_a"]} for p in pairs]
    return [{"mm": m, "um": u} for m, u in zip(mm, um)]


@torch.no_grad()
def hidden_means(model, items, modality: str, layer: int, device: str, batch_size: int):
    reps = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        batch = collate_plain(chunk, model.processor, types.SimpleNamespace(max_length=model.max_length))
        mm, um = batch["mm"], batch["um"]
        if modality == "it":
            ids, attn, pixel, _ = mm
        else:
            ids, attn, _, _ = um
            pixel = None
        ids, attn = ids.to(device), attn.to(device)
        if pixel is not None:
            pixel = pixel.to(device)
        out = model.model(input_ids=ids, attention_mask=attn, pixel_values=pixel,
                          output_hidden_states=True, use_cache=False)
        h = out.hidden_states[layer].float()
        mask = attn.unsqueeze(-1).float()
        reps.append(((h * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu())
    return torch.cat(reps, dim=0)


class Wrapped:
    def __init__(self, model, processor, max_length):
        self.model = model
        self.processor = processor
        self.max_length = max_length


def main():
    parser = argparse.ArgumentParser(description="CKA: base vs unlearned, IT vs PT")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", required=True, help="name=path,...")
    parser.add_argument("--data_split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--forget_split", default="forget_5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    items = probe_pairs(args.data_split_dir, args.forget_split, args.n, processor,
                        args.max_length, args.seed)

    base = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device).eval()
    wrapped = Wrapped(base, processor, args.max_length)

    reps = {"base": {"it": hidden_means(wrapped, items, "it", args.layer, args.device, args.batch_size),
                     "pt": hidden_means(wrapped, items, "pt", args.layer, args.device, args.batch_size)}}
    print("base reps done", flush=True)

    adapters = [it.split("=", 1) for it in args.adapters.split(",")]
    peft = PeftModel.from_pretrained(base, adapters[0][1], adapter_name=adapters[0][0])
    peft.eval()
    for name, path in adapters:
        if name != adapters[0][0]:
            peft.load_adapter(path, adapter_name=name)
        peft.set_adapter(name)
        w = Wrapped(peft, processor, args.max_length)
        reps[name] = {"it": hidden_means(w, items, "it", args.layer, args.device, args.batch_size),
                      "pt": hidden_means(w, items, "pt", args.layer, args.device, args.batch_size)}
        print(f"{name} reps done", flush=True)

    result = {"layer": args.layer, "n": len(items), "cka": {}}
    names = list(reps)
    for name in names:
        result["cka"][name] = {
            "it_vs_pt": cka(reps[name]["it"], reps[name]["pt"]),
            "it_vs_base": cka(reps[name]["it"], reps["base"]["it"]),
            "pt_vs_base": cka(reps[name]["pt"], reps["base"]["pt"]),
        }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'model':>10} {'IT-vs-PT':>9} {'IT-vs-base':>11} {'PT-vs-base':>11}")
    for name, v in result["cka"].items():
        print(f"{name:>10} {v['it_vs_pt']:>9.4f} {v['it_vs_base']:>11.4f} {v['pt_vs_base']:>11.4f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
