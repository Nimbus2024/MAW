#!/usr/bin/env python3
"""E2: 两个 LoRA adapter 的 ΔW 几何对比 (逐层/逐模块余弦、范数、主方向)。

ΔW = scaling * (B @ A), scaling = lora_alpha / r。
输入: 两个 PEFT adapter 目录 (adapter_model.safetensors + adapter_config.json)。
输出: JSON (global/per-layer/per-module cosine 与范数) + 控制台摘要。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_adapter(directory: Path):
    cfg = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    weights = load_file(str(directory / "adapter_model.safetensors"))
    scaling = float(cfg.get("lora_alpha", 1.0)) / float(cfg.get("r", 1))
    deltas = {}
    keys = {}
    for name, tensor in weights.items():
        match = re.match(r"(.+)\.lora_(A|B)\.weight$", name)
        if not match:
            continue
        module = match.group(1)
        keys.setdefault(module, {})[match.group(2)] = tensor.float()
    for module, pair in keys.items():
        if "A" in pair and "B" in pair:
            deltas[module] = scaling * (pair["B"] @ pair["A"])
    return deltas, scaling


def layer_of(module: str):
    match = re.search(r"layers\.(\d+)\.", module)
    if match:
        return int(match.group(1))
    if "multi_modal_projector" in module:
        return "projector"
    if "vision" in module:
        return "vision"
    return "other"


def cosine(a: torch.Tensor, b: torch.Tensor):
    a = a.flatten()
    b = b.flatten()
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return None
    return float((a @ b) / (na * nb))


def compare(deltas_a, deltas_b):
    shared = sorted(set(deltas_a) & set(deltas_b))
    per_module = {}
    for module in shared:
        da, db = deltas_a[module], deltas_b[module]
        per_module[module] = {
            "cos": cosine(da, db),
            "norm_a": float(da.norm()),
            "norm_b": float(db.norm()),
            "shape": list(da.shape),
        }
    by_layer = {}
    for module in shared:
        layer = layer_of(module)
        by_layer.setdefault(layer, []).append(module)
    per_layer = {}
    for layer, modules in sorted(by_layer.items(), key=lambda kv: str(kv[0])):
        ca = torch.cat([deltas_a[m].flatten() for m in modules])
        cb = torch.cat([deltas_b[m].flatten() for m in modules])
        per_layer[str(layer)] = {
            "cos": cosine(ca, cb),
            "norm_a": float(ca.norm()),
            "norm_b": float(cb.norm()),
            "n_modules": len(modules),
        }
    ga = torch.cat([deltas_a[m].flatten() for m in shared])
    gb = torch.cat([deltas_b[m].flatten() for m in shared])
    global_cos = cosine(ga, gb)
    return {
        "n_modules": len(shared),
        "missing_in_b": sorted(set(deltas_a) - set(deltas_b)),
        "missing_in_a": sorted(set(deltas_b) - set(deltas_a)),
        "global": {"cos": global_cos, "norm_a": float(ga.norm()), "norm_b": float(gb.norm())},
        "per_layer": per_layer,
        "per_module": per_module,
    }


def main():
    parser = argparse.ArgumentParser(description="E2: adapter ΔW 几何对比")
    parser.add_argument("--adapter_a", required=True)
    parser.add_argument("--adapter_b", required=True)
    parser.add_argument("--label_a", default="A")
    parser.add_argument("--label_b", default="B")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    da, sa = load_adapter(Path(args.adapter_a))
    db, sb = load_adapter(Path(args.adapter_b))
    result = {
        "adapter_a": {"path": args.adapter_a, "label": args.label_a, "scaling": sa, "n_modules": len(da)},
        "adapter_b": {"path": args.adapter_b, "label": args.label_b, "scaling": sb, "n_modules": len(db)},
        "comparison": compare(da, db),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    comp = result["comparison"]
    print(f"global cos={comp['global']['cos']:.4f} "
          f"norm_a={comp['global']['norm_a']:.3f} norm_b={comp['global']['norm_b']:.3f}")
    for layer, vals in comp["per_layer"].items():
        cos = vals["cos"]
        print(f"  layer {layer}: cos={'NA' if cos is None else f'{cos:+.4f}'} "
              f"n_mod={vals['n_modules']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
