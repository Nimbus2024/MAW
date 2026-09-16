#!/usr/bin/env python3
"""图像 token knockout: 逐层把 image token 的层输入 hidden 置零, 测 IT 答案 CE 的上升,
定位"图像信息被使用"的层 (跨模态信息流的简化因果版)。
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


def pairs_of(data_split_dir, split, n, seed=42):
    ps = build_pairs(pd.read_parquet(Path(data_split_dir) / split / "train-00000-of-00001.parquet"))
    order = torch.randperm(len(ps), generator=torch.Generator().manual_seed(seed)).tolist()
    return [ps[i] for i in order[:n]]


@torch.no_grad()
def ce_image_knockout(model, item, layer, device, processor, max_length, n_img):
    batch = build_batch([item], "mm", processor, max_length)
    ids, attn, pixel, labels = batch_to_device(batch, "mm", device)
    img_tok = model.config.image_token_index
    pos = int((ids[0] == img_tok).nonzero(as_tuple=True)[0][0].item())

    def hook(module, args):
        x = args[0].clone()
        end = min(x.shape[1], pos + n_img)
        x[:, pos:end] = 0
        return (x,)

    h = get_layers(model)[layer].register_forward_pre_hook(hook)
    try:
        return float(model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels).loss)
    finally:
        h.remove()


@torch.no_grad()
def ce_plain(model, item, device, processor, max_length):
    batch = build_batch([item], "mm", processor, max_length)
    ids, attn, pixel, labels = batch_to_device(batch, "mm", device)
    return float(model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels).loss)


def main():
    parser = argparse.ArgumentParser(description="图像 token 逐层 knockout")
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", default="")
    parser.add_argument("--data_split_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,28,31")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--n_img", type=int, default=576)
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
    pairs = pairs_of(args.data_split_dir, args.forget_split, args.n, args.seed)
    items = [{"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]} for p in pairs]

    result = {"layers": layers, "n_img": args.n_img, "models": {}}
    for item in [""] + [a for a in args.adapters.split(",") if a]:
        if item == "":
            name = "base"
        else:
            name, path = item.split("=", 1)
            ab, scaling = load_adapter_ab(Path(path))
            ab = {norm_name(k): v for k, v in ab.items()}
            apply_delta(modules, ab, scaling, +1.0)
        plain = sum(ce_plain(model, it, args.device, processor, args.max_length) for it in items) / len(items)
        knocked = {}
        for L in layers:
            knocked[str(L)] = sum(ce_image_knockout(model, it, L, args.device, processor, args.max_length, args.n_img)
                                  for it in items) / len(items)
        result["models"][name] = {"plain_ce": plain,
                                  "knockout_ce": knocked,
                                  "delta": {L: v - plain for L, v in knocked.items()}}
        if item != "":
            apply_delta(modules, ab, scaling, -1.0)
        print(f"{name}: plain={plain:.4f}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("layer  base_delta  joint_delta")
    for L in layers:
        j = result["models"].get("joint", {}).get("delta", {}).get(str(L))
        print(f"{L:>5} {result['models']['base']['delta'][str(L)]:>11.4f} {j}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
