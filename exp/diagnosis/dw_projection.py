#!/usr/bin/env python3
"""P0-P2: 把各变体 adapter 的 ΔW 分解到 mm-only / um-only 两个参照方向上。

对每个 target:
  c_mm = <dW, dW_mm> / ||dW_mm||^2 ,  c_um = <dW, dW_um> / ||dW_um||^2
  cos_mm, cos_um, ||dW||
逐模块用迹技巧累积 (不物化整段 dW):
  <dW1,dW2> = s1*s2 * Σ_mod trace(A1^T (B1^T B2) A2)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .adapter_geometry import load_adapter_ab


def norm_name(name: str) -> str:
    return name[name.index("language_model."):] if "language_model." in name else name


def load(path):
    ab, scaling = load_adapter_ab(Path(path))
    return {norm_name(k): v for k, v in ab.items()}, float(scaling)


def inner(ab1, s1, ab2, s2):
    tot = 0.0
    for n in ab1:
        if n not in ab2:
            continue
        A1, B1 = ab1[n]
        A2, B2 = ab2[n]
        M = B1.t() @ B2
        tot += float((A1 * (M @ A2)).sum())
    return s1 * s2 * tot


def main():
    ap = argparse.ArgumentParser(description="ΔW 在 mm/um 参照方向上的分解")
    ap.add_argument("--ref_mm", required=True)
    ap.add_argument("--ref_um", required=True)
    ap.add_argument("--targets", required=True, help="tag=path,...")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    mm, sm = load(args.ref_mm)
    um, su = load(args.ref_um)
    n_mm2 = inner(mm, sm, mm, sm)
    n_um2 = inner(um, su, um, su)
    x_mm = inner(mm, sm, um, su)

    res = {"ref": {"norm_mm": n_mm2 ** 0.5, "norm_um": n_um2 ** 0.5,
                   "cos_mm_um": x_mm / ((n_mm2 * n_um2) ** 0.5) if n_mm2 > 0 and n_um2 > 0 else None},
           "targets": {}}
    for item in args.targets.split(","):
        tag, path = item.split("=", 1)
        ab, s = load(path)
        n2 = inner(ab, s, ab, s)
        e = {"norm": n2 ** 0.5,
             "c_mm": inner(ab, s, mm, sm) / n_mm2 if n_mm2 > 0 else None,
             "c_um": inner(ab, s, um, su) / n_um2 if n_um2 > 0 else None}
        d = n2 ** 0.5
        e["cos_mm"] = (inner(ab, s, mm, sm) / (d * n_mm2 ** 0.5)) if d > 0 and n_mm2 > 0 else None
        e["cos_um"] = (inner(ab, s, um, su) / (d * n_um2 ** 0.5)) if d > 0 and n_um2 > 0 else None
        res["targets"][tag] = e
        print(f"{tag:>16}: norm={e['norm']:.4f} c_mm={e['c_mm']:+.4f} c_um={e['c_um']:+.4f} "
              f"cos_mm={e['cos_mm']:+.3f} cos_um={e['cos_um']:+.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ref: norm_mm={res['ref']['norm_mm']:.4f} norm_um={res['ref']['norm_um']:.4f} "
          f"cos(mm,um)={res['ref']['cos_mm_um']:+.3f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
