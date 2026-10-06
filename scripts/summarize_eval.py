#!/usr/bin/env python3
"""汇总 p02_eval_* 的 UMU-bench 结果, 并与 CE 层的"效应实现比例"并列。

用法: python scripts/summarize_eval.py [--root results/_analysis] [--tags a,b,c]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def load(root, tag):
    f = glob.glob(str(Path(root) / f"p02_eval_{tag}" / f"{tag}_final_evaluation_results.json"))
    if not f:
        return None
    d = json.load(open(f[0]))
    out = {}
    scope_map = {"Forget Set Results": "forget",
                 "Retain Set (shared dataset) Results": "retain",
                 "Retain Set (real person) Results": "real"}
    for k, short in scope_map.items():
        s = d[k]
        fl = s["fill_in_the_blank"]; cl = s["classification"]; ge = s["generation"]
        out[short] = {
            "fill": (fl["image_textual_accuracy"], fl["pure_text_accuracy"]),
            "cls": (cl["Image-Textual Question Accuracy"], cl["Pure Text Question Accuracy"]),
            "gen": (ge["Average ROUGE-L (Image_Textual)"], ge["Average ROUGE-L (Pure_Text)"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/_analysis")
    ap.add_argument("--tags", default="origin,refmm,refum,p0clip1.0,p1bal,merged_mm_um")
    args = ap.parse_args()
    tags = args.tags.split(",")
    R = {t: load(args.root, t) for t in tags}

    for scope, tasks in (("forget", ("fill", "cls", "gen")), ("retain", ("fill", "cls")), ("real", ("cls",))):
        print(f"\n=== {scope} ===")
        print(f"{'model':>16} | " + " | ".join(f"{t} (IT/PT)".rjust(16) for t in tasks))
        for t in tags:
            if not R[t]:
                print(f"{t:>16} | (no result)")
                continue
            cells = []
            for task in tasks:
                a, b = R[t][scope][task]
                cells.append(f"{a:.3f}/{b:.3f}".rjust(16))
            print(f"{t:>16} | " + " | ".join(cells))

if __name__ == "__main__":
    main()
