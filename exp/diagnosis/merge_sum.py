#!/usr/bin/env python3
"""P3: 把 base + α·ΔW_mm + β·ΔW_um 合并成一个**完整模型** (供 bench eval 用 `--pretrain`)。

与 dw_dose.py 的施加方式一致 (apply_delta), 故 CE 行为应可复现。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import LlavaForConditionalGeneration

from .adapter_geometry import load_adapter_ab
from .patching import apply_delta, norm_name, target_modules


def load_ab(path):
    ab, scaling = load_adapter_ab(Path(path))
    return {norm_name(k): v for k, v in ab.items()}, float(scaling)


def main():
    ap = argparse.ArgumentParser(description="合并 ΔW 组合为完整模型")
    ap.add_argument("--base", required=True)
    ap.add_argument("--ref_mm", required=True)
    ap.add_argument("--ref_um", required=True)
    ap.add_argument("--scale_mm", type=float, default=1.0)
    ap.add_argument("--scale_um", type=float, default=1.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
    modules = target_modules(model)
    mm, sm = load_ab(args.ref_mm)
    um, su = load_ab(args.ref_um)
    if args.scale_mm != 0:
        apply_delta(modules, mm, sm * args.scale_mm, +1.0)
    if args.scale_um != 0:
        apply_delta(modules, um, su * args.scale_um, +1.0)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    print(f"saved merged model (mm*{args.scale_mm} + um*{args.scale_um}) -> {out}")


if __name__ == "__main__":
    main()
