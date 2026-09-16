#!/usr/bin/env python3
"""Task-arithmetic 分解: 检验 ΔW_joint ≈ ΔW_mm + ΔW_um, 以及三者的子空间重叠。

逐模块累积 (避免整段物化):
  ||ΔW_joint - ΔW_mm - ΔW_um||^2 / ||ΔW_joint||^2   (additivity residual)
  cos(ΔW_mm, ΔW_um), cos(ΔW_joint, ΔW_mm), cos(ΔW_joint, ΔW_um)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .adapter_geometry import load_adapter_ab


def norm_name(name: str) -> str:
    return name[name.index("language_model."):] if "language_model." in name else name


def load(name_path):
    name, path = name_path.split("=", 1)
    ab, scaling = load_adapter_ab(Path(path))
    return name, {norm_name(k): v for k, v in ab.items()}, scaling


def main():
    parser = argparse.ArgumentParser(description="task-arithmetic 分解 (mm/um/joint)")
    parser.add_argument("--mm", required=True)
    parser.add_argument("--um", required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    _, mm, sm = load(f"mm={args.mm}")
    _, um, su = load(f"um={args.um}")
    _, joint, sj = load(f"joint={args.joint}")
    keys = sorted(set(mm) & set(um) & set(joint))

    n_j2 = n_m2 = n_u2 = n_res2 = 0.0
    dot_mu = dot_jm = dot_ju = 0.0
    for k in keys:
        Am, Bm = mm[k]
        Au, Bu = um[k]
        Aj, Bj = joint[k]
        dm = sm * (Bm @ Am)
        du = su * (Bu @ Au)
        dj = sj * (Bj @ Aj)
        res = dj - dm - du
        n_j2 += float((dj * dj).sum())
        n_m2 += float((dm * dm).sum())
        n_u2 += float((du * du).sum())
        n_res2 += float((res * res).sum())
        dot_mu += float((dm * du).sum())
        dot_jm += float((dj * dm).sum())
        dot_ju += float((dj * du).sum())

    nj, nm, nu = math.sqrt(n_j2), math.sqrt(n_m2), math.sqrt(n_u2)
    result = {
        "n_modules": len(keys),
        "norm_joint": nj, "norm_mm": nm, "norm_um": nu,
        "additivity_residual_frac": n_res2 / n_j2 if n_j2 > 0 else None,
        "cos_mm_um": dot_mu / (nm * nu) if nm > 0 and nu > 0 else None,
        "cos_joint_mm": dot_jm / (nj * nm) if nj > 0 and nm > 0 else None,
        "cos_joint_um": dot_ju / (nj * nu) if nj > 0 and nu > 0 else None,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
