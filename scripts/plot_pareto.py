#!/usr/bin/env python3
"""bench 层主指标可视化 (a=AllErr, d=AllAcc, oe=1-d-a)。

图:
  fig_p8_pareto_forget_retain   forget-a × retain(shared)-d   (Fill / Cls)
  fig_p9_pareto_forget_real     forget-a × real-d             (Fill / Cls)
  fig_p10_oe_vs_forgeta         oe × forget-a, 3x2 网格 (set × task)

非支配前沿: 两目标都"越大越好"时标出非支配点并画阶梯线。
坐标相同的模型会合并成一个点并联合标注（如 "joint(P0) = clip0.2"）。
数据: results/_analysis/p02_eval_<tag>/<tag>_final_evaluation_results.json
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SETS = {"forget": "Forget Set Results",
        "retain": "Retain Set (shared dataset) Results",
        "real": "Retain Set (real person) Results"}
TASKS = ("fill", "cls")
LABELS = {"origin": "origin", "refmm": "refmm", "refum": "refum", "p0clip1.0": "joint(P0)",
          "p1bal": "gradbal(P1)", "merged_mm_um": "merged(P3)", "clip0.2": "clip0.2",
          "clip5.0": "clip5", "clip100.0": "clip100",
          "noclip_gs0.04": "gs0.04", "noclip_gs0.20": "gs0.20"}
MODELS = list(LABELS)
OFFSETS = [(6, 4), (6, -12), (-58, 4), (-58, -12), (6, 14), (-58, 14)]


def save(fig, outdir, name):
    fig.savefig(Path(outdir) / f"{name}.svg")
    fig.savefig(Path(outdir) / f"{name}.png", dpi=140)


def block(d, scope, task):
    s = d[SETS[scope]]
    return s["fill_in_the_blank"] if task == "fill" else s["classification"]


def load(root, tag):
    f = glob.glob(str(Path(root) / f"p02_eval_{tag}" / f"{tag}_final_evaluation_results.json"))
    if not f:
        return None
    d = json.load(open(f[0]))
    out = {}
    for scope in SETS:
        for task in TASKS:
            b = block(d, scope, task)
            if task == "fill":
                it, pt = b["image_textual_accuracy"] / 100, b["pure_text_accuracy"] / 100
            else:
                it, pt = b["Image-Textual Question Accuracy"] / 100, b["Pure Text Question Accuracy"] / 100
            A, E = b["All Modal Question Accuracy"] / 100, b["All Modal Question Error"] / 100
            out[(scope, task)] = {"it": it, "pt": pt, "a": E, "d": A, "oe": 1 - A - E}
    return out


def groups(pts):
    g = defaultdict(list)
    for x, y, n in pts:
        g[(round(x, 3), round(y, 3))].append(n)
    return sorted(g.items())


def nondominated(pts):
    keys = []
    for (x, y), _ in groups(pts):
        if not any((x2 >= x and y2 >= y and (x2 > x or y2 > y))
                   for (x2, y2), _ in groups(pts) if (x2, y2) != (x, y)):
            keys.append((x, y))
    return sorted(keys)


def draw(ax, pts, xlabel, ylabel, title, frontier=True):
    nds = nondominated(pts) if frontier else []
    for i, ((x, y), names) in enumerate(groups(pts)):
        base = "origin" in names
        nd = (x, y) in nds
        ax.scatter([x], [y], s=110 if base else 70, marker="*" if base else "o",
                   facecolors="gold" if base else ("none" if nd else "tab:blue"),
                   edgecolors="crimson" if nd else "black", linewidths=1.7, zorder=3)
        ax.annotate(" = ".join(LABELS[n] for n in names), (x, y), textcoords="offset points",
                    xytext=OFFSETS[i % len(OFFSETS)], fontsize=7)
    if frontier and len(nds) >= 2:
        sx, sy = [nds[0][0]], [nds[0][1]]
        for (x1, y1), (x2, y2) in zip(nds, nds[1:]):
            sx += [x2, x2]; sy += [y1, y2]
        ax.plot(sx, sy, "--", color="crimson", lw=1.3, zorder=2, label="non-dominated frontier")
        ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.margins(0.13); ax.grid(alpha=.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="../product/results/_analysis")
    ap.add_argument("--outdir", default="../references/think/figs")
    args = ap.parse_args()
    root, out = Path(args.eval_root), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    D = {t: load(root, t) for t in MODELS}
    D = {t: v for t, v in D.items() if v}
    print(f"{'model':>10} | " + " | ".join(f"{s}-{k} a/d/oe" for s in ("for", "ret", "rea") for k in TASKS))
    for t, v in D.items():
        print(f"{t:>10} | " + " | ".join(
            f"{v[(s, k)]['a']:.2f}/{v[(s, k)]['d']:.2f}/{v[(s, k)]['oe']:.2f}"
            for s in ("forget", "retain", "real") for k in TASKS))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, task in zip(axes, TASKS):
        pts = [(D[t][("forget", task)]["a"], D[t][("retain", task)]["d"], t) for t in D]
        draw(ax, pts, f"forget-{task} AllErr (higher = both forgotten)",
             f"retain-{task} AllAcc (higher = both kept)", f"P1  forget-a vs retain-d  ({task})")
    fig.suptitle("Pareto: consistent forgetting vs consistent retention", fontsize=12)
    fig.tight_layout(); save(fig, out, "fig_p8_pareto_forget_retain"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, task in zip(axes, TASKS):
        pts = [(D[t][("forget", task)]["a"], D[t][("real", task)]["d"], t) for t in D]
        draw(ax, pts, f"forget-{task} AllErr (higher = both forgotten)",
             f"real-{task} AllAcc (higher = both kept)", f"P2  forget-a vs real-d  ({task})")
    fig.suptitle("Pareto: consistent forgetting vs real-person retention", fontsize=12)
    fig.tight_layout(); save(fig, out, "fig_p9_pareto_forget_real"); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    for r, scope in enumerate(("forget", "retain", "real")):
        for c, task in enumerate(TASKS):
            ax = axes[r, c]
            pts = [(D[t][("forget", task)]["a"], D[t][(scope, task)]["oe"], t) for t in D]
            draw(ax, pts, f"forget-{task} AllErr (higher = more forgotten)",
                 f"{scope}-{task} oe (lower = more balanced)", f"{scope} / {task}", frontier=False)
    fig.suptitle("oe vs forget-a  (ideal: bottom-right)", fontsize=13)
    fig.tight_layout(); save(fig, out, "fig_p10_oe_vs_forgeta"); plt.close(fig)

    print("saved 3 figures to", out)


if __name__ == "__main__":
    main()
