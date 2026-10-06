#!/usr/bin/env python3
"""bench 层主指标可视化 (a=AllErr, d=AllAcc, oe=1-d-a)。

图:
  fig_p8_pareto_forget_retain   forget-a × retain(shared)-d   (Fill / Cls)
  fig_p9_pareto_forget_real     forget-a × real-d             (Fill / Cls)
  fig_p10_oe_vs_forgeta         oe × forget-a, 3x2 网格 (set × task)

标注方式: 每个模型一种固定颜色, 图例统一在图上; 坐标完全重合的点做小抖动避免遮挡。
非支配前沿: 两目标都"越大越好"时标出非支配点并画阶梯线。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SETS = {"forget": "Forget Set Results",
        "retain": "Retain Set (shared dataset) Results",
        "real": "Retain Set (real person) Results"}
TASKS = ("fill", "cls")
LABELS = {"origin": "origin", "refmm": "refmm", "refum": "refum",
          "clip0.2": "clip0.2", "p0clip1.0": "joint(P0)=clip1.0", "clip5.0": "clip5",
          "clip100.0": "clip100", "p1bal": "gradbal(P1)", "merged_mm_um": "merged(P3)",
          "noclip_gs0.04": "gs0.04", "noclip_gs0.20": "gs0.20"}
MODELS = ["origin", "refmm", "refum", "clip0.2", "p0clip1.0", "clip5.0", "clip100.0",
          "p1bal", "merged_mm_um", "noclip_gs0.04", "noclip_gs0.20"]
COLORS = {"origin": "#FFD700",
          "refmm": "#1f77b4", "refum": "#ff7f0e",
          "clip0.2": "#7f0000", "p0clip1.0": "#c62828", "clip5.0": "#ef5350", "clip100.0": "#ffb3b0",
          "p1bal": "#2ca02c", "merged_mm_um": "#9467bd",
          "noclip_gs0.04": "#424242", "noclip_gs0.20": "#9e9e9e"}


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


def jitter(pts, r=0.004):
    g = defaultdict(list)
    for x, y, n in pts:
        g[(round(x, 4), round(y, 4))].append((x, y, n))
    out = []
    for items in g.values():
        if len(items) == 1:
            out.append(items[0])
        else:
            for i, (x, y, n) in enumerate(items):
                ang = 2 * math.pi * i / len(items)
                out.append((x + r * math.cos(ang), y + r * math.sin(ang), n))
    return out


def nondominated(pts, tol=5e-3):
    keys = [(x, y) for x, y, _ in pts]
    nd = []
    for i, (x, y) in enumerate(keys):
        if any((x2 >= x - tol and y2 >= y - tol and (x2 > x + tol or y2 > y + tol))
               for j, (x2, y2) in enumerate(keys) if j != i):
            continue
        nd.append((x, y))
    return sorted(nd)


def draw(ax, pts, xlabel, ylabel, title, frontier=True):
    jp = jitter(pts)
    nds = nondominated(jp) if frontier else []
    ndset = set(nds)
    handles, labels = [], []
    for x, y, n in jp:
        base = n == "origin"
        nd = (round(x, 4), round(y, 4)) in ndset or any(abs(x - a) < 2e-4 and abs(y - b) < 2e-4 for a, b in nds)
        h = ax.scatter([x], [y], s=170 if base else 62, marker="*" if base else "o",
                       color=COLORS[n], edgecolors="black" if nd else "k",
                       linewidths=1.6 if nd else 0.6, zorder=3 + (1 if nd else 0))
        handles.append(h); labels.append(LABELS[n])
    if frontier and len(nds) >= 2:
        sx, sy = [nds[0][0]], [nds[0][1]]
        for (x1, y1), (x2, y2) in zip(nds, nds[1:]):
            sx += [x2, x2]; sy += [y1, y2]
        ax.plot(sx, sy, "--", color="crimson", lw=1.3, zorder=2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.margins(0.12); ax.grid(alpha=.3)
    return handles, labels


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
        h, l = draw(ax, pts, f"forget-{task} AllErr (higher = both forgotten)",
                    f"retain-{task} AllAcc (higher = both kept)", f"P1  forget-a vs retain-d  ({task})")
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("Pareto: consistent forgetting vs consistent retention   "
                 "[reds = clip max_norm 0.2/1/5/100; grey = no-clip grad_scale; dashed = frontier]",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96)); save(fig, out, "fig_p8_pareto_forget_retain"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, task in zip(axes, TASKS):
        pts = [(D[t][("forget", task)]["a"], D[t][("real", task)]["d"], t) for t in D]
        h, l = draw(ax, pts, f"forget-{task} AllErr (higher = both forgotten)",
                    f"real-{task} AllAcc (higher = both kept)", f"P2  forget-a vs real-d  ({task})")
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("Pareto: consistent forgetting vs real-person retention   "
                 "[reds = clip max_norm 0.2/1/5/100; grey = no-clip grad_scale; dashed = frontier]",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96)); save(fig, out, "fig_p9_pareto_forget_real"); plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    h = l = None
    for r, scope in enumerate(("forget", "retain", "real")):
        for c, task in enumerate(TASKS):
            ax = axes[r, c]
            pts = [(D[t][("forget", task)]["a"], D[t][(scope, task)]["oe"], t) for t in D]
            h, l = draw(ax, pts, f"forget-{task} AllErr (higher = more forgotten)",
                        f"{scope}-{task} oe (lower = more balanced)", f"{scope} / {task}", frontier=False)
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("oe vs forget-a  (ideal: bottom-right)   "
                 "[reds = clip max_norm 0.2/1/5/100; grey = no-clip grad_scale]", fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.975)); save(fig, out, "fig_p10_oe_vs_forgeta"); plt.close(fig)

    print("saved 3 figures to", out)


if __name__ == "__main__":
    main()
