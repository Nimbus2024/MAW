#!/usr/bin/env python3
"""P3/P4: 对 base 施加线性组合 α·ΔW_mm + β·ΔW_um, 测 IT/PT 问题平均 CE。

P3 (叠加性 N2): 比较 B(ΔW_mm+ΔW_um) 与 B(ΔW_mm)+B(ΔW_um)(即 ΔCE 是否可加), 并与 B(ΔW_joint) 对照。
P4 (局部线性 N1): 固定 β=0(或 α=0), 扫 λ 得 B(λ·ΔW_mm) 曲线, 看 CE 是否随 λ 线性。

configs 语法: "alpha,beta;alpha,beta;..."  (另用 --joint 单测 joint adapter)。
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
from .ce_breakdown import agg, ce_stats
from .patching import apply_delta, norm_name, target_modules


def load_ab(path):
    ab, scaling = load_adapter_ab(Path(path))
    return {norm_name(k): v for k, v in ab.items()}, float(scaling)


@torch.no_grad()
def measure(model, processor, device, split_dir, max_length, batch_size):
    pairs = build_pairs(pd.read_parquet(Path(split_dir) / "forget_5" / "train-00000-of-00001.parquet"))
    it = [{"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]} for p in pairs]
    pt = [{"question": p["um_q"], "answer": p["um_a"]} for p in pairs]
    ce_i, nt_i = ce_stats(model, it, "mm", processor, device, max_length, batch_size)
    ce_p, nt_p = ce_stats(model, pt, "um", processor, device, max_length, batch_size)
    return {"it": agg(ce_i, nt_i), "pt": agg(ce_p, nt_p)}


def main():
    ap = argparse.ArgumentParser(description="α·ΔW_mm + β·ΔW_um 的剂量-效应")
    ap.add_argument("--base", required=True)
    ap.add_argument("--ref_mm", required=True)
    ap.add_argument("--ref_um", required=True)
    ap.add_argument("--joint", default=None)
    ap.add_argument("--configs", default="0,0;1,0;0,1;1,1")
    ap.add_argument("--data_split_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device).eval()
    model.config.use_cache = False
    modules = target_modules(model)
    mm, sm = load_ab(args.ref_mm)
    um, su = load_ab(args.ref_um)

    res = {"configs": {}, "joint": None}
    for spec in args.configs.split(";"):
        a, b = (float(x) for x in spec.split(","))
        key = f"{a:g},{b:g}"
        if a != 0:
            apply_delta(modules, mm, sm * a, +1.0)
        if b != 0:
            apply_delta(modules, um, su * b, +1.0)
        res["configs"][key] = measure(model, processor, args.device, args.data_split_dir,
                                      args.max_length, args.batch_size)
        if b != 0:
            apply_delta(modules, um, su * b, -1.0)
        if a != 0:
            apply_delta(modules, mm, sm * a, -1.0)
        e = res["configs"][key]
        print(f"α={a:g} β={b:g}: IT={e['it']['mean']:.5f} PT={e['pt']['mean']:.5f}", flush=True)

    if args.joint:
        jb, sj = load_ab(args.joint)
        apply_delta(modules, jb, sj, +1.0)
        res["joint"] = measure(model, processor, args.device, args.data_split_dir,
                               args.max_length, args.batch_size)
        apply_delta(modules, jb, sj, -1.0)
        print(f"joint: IT={res['joint']['it']['mean']:.5f} PT={res['joint']['pt']['mean']:.5f}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
