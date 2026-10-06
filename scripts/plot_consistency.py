#!/usr/bin/env python3
"""一致性 / 效率图 (bench 主指标).

主指标: a = AllErr = P(两模态都错); d = AllAcc = P(两模态都对); oe = 1-a-d = P(恰一侧错).
  forget = Forget Set;  retain = Retain Set (shared dataset);  real = Retain Set (real person).
  lr5x 崩坏项已剔除; origin 在所有图里作为参考点 (金色★ + 金色等 oe 线).

两类图:
  fig_p11_isooe_grid   3x2 = (forget / retain / real) x (fill / cls)
                       x=d, y=a, 每个点画 a+d=c 的等 oe 直线 (oe=1-a-d)。
                       同一 d 竖线上比 a: a 越高 oe 越低 (效率越高)。
  fig_p12_pareto_grid  2x2 = task(fill/cls) x (forget-a x retain-d / forget-a x real-d)
                       两轴都越大越好 -> 右上角 Pareto。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_pareto import (  # noqa: E402
    SETS, TASKS, LABELS, MODELS, COLORS, load, save,
)


def _limits(vals, pad):
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1.0
    return lo - pad, hi + pad


def _isooe_label_pos(x0, x1, y0, y1, c):
    """等 oe 线 a = c - d 在面板框内的一个标注位置 (优先右边缘, 否则上边缘)。"""
    a_right = c - x1
    if y0 <= a_right <= y1:
        return x1, a_right
    d_top = c - y1
    if x0 <= d_top <= x1:
        return d_top, y1
    return x1, y0


def draw_ad(ax, pts, scope, task, good, xpad, ypad, step):
    """pts: [(d_pct, a_pct, name)]; 每个点画其等 oe 直线; origin 线加粗。"""
    ds = [p[0] for p in pts]
    as_ = [p[1] for p in pts]
    x0, x1 = _limits(ds, xpad)
    y0, y1 = _limits(as_, ypad)

    oes = [100 - (d + a) for d, a, _ in pts]
    oe = int(math.ceil(min(oes) / step)) * step
    while oe <= max(oes) + 0.5:
        c = 100 - oe
        lx, ly = _isooe_label_pos(x0, x1, y0, y1, c)
        ax.plot([x0, x1], [c - x0, c - x1], ls="--", lw=0.6,
                color="#cccccc", alpha=0.9, zorder=1)
        ax.text(lx + 0.3, ly, f"oe={oe}", fontsize=6, color="#999999",
                va="center", ha="left", zorder=1)
        oe += step

    lx, ly = _isooe_label_pos(x0, x1, y0, y1, 100)
    ax.plot([x0, x1], [100 - x0, 100 - x1], ls="-", lw=1.1, color="#777777",
            alpha=0.95, zorder=1)
    ax.text(lx + 0.3, ly, "oe=0", fontsize=6.5, color="#555555", va="center",
            ha="left", zorder=1)

    seen = set()
    for d, a, n in pts:
        c = d + a
        key = round(c, 3)
        if key not in seen:
            seen.add(key)
            col = COLORS[n]
            ax.plot([x0, x1], [c - x0, c - x1], lw=2.0 if n == "origin" else 1.0,
                    color=col, alpha=0.85 if n == "origin" else 0.30,
                    ls="-" if n == "origin" else "--", zorder=2)

    handles, labels = [], []
    for d, a, n in pts:
        base = n == "origin"
        h = ax.scatter([d], [a], s=190 if base else 66,
                       marker="*" if base else "o", color=COLORS[n],
                       edgecolors="k", linewidths=0.8, zorder=5)
        handles.append(h)
        labels.append(LABELS[n])

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("d = AllAcc  (both correct)")
    ax.set_ylabel("a = AllErr  (both wrong)")
    ax.set_title(f"{scope} / {task}\n{good}", fontsize=9)
    ax.grid(alpha=0.25)
    return handles, labels


def draw_pareto(ax, pts, xlabel, ylabel, title):
    """pts: [(x, y, name)]; 两轴都越大越好 -> 右上 Pareto。"""
    jp = list(pts)
    keys = sorted((x, y) for x, y, _ in jp)
    nd = []
    for x, y in keys:
        if not any(x2 >= x - 5e-3 and y2 >= y - 5e-3 and (x2 > x + 5e-3 or y2 > y + 5e-3)
                   for x2, y2 in keys if (x2, y2) != (x, y)):
            nd.append((x, y))
    handles, labels = [], []
    for x, y, n in jp:
        base = n == "origin"
        is_nd = any(abs(x - a) < 2e-4 and abs(y - b) < 2e-4 for a, b in nd)
        h = ax.scatter([x], [y], s=190 if base else 66,
                       marker="*" if base else "o", color=COLORS[n],
                       edgecolors="crimson" if is_nd else "k",
                       linewidths=1.6 if is_nd else 0.6, zorder=4 if is_nd else 3)
        handles.append(h)
        labels.append(LABELS[n])
        if base:
            ax.annotate("origin", (x, y), textcoords="offset points",
                        xytext=(6, -10), fontsize=8, color="#7a6000", zorder=6)
    if len(nd) >= 2:
        nd = sorted(nd)
        sx, sy = [nd[0][0]], [nd[0][1]]
        for (x1, y1), (x2, y2) in zip(nd, nd[1:]):
            sx += [x2, x2]
            sy += [y1, y2]
        ax.plot(sx, sy, "--", color="crimson", lw=1.3, zorder=1)

    # origin 参考十字线
    ox, oy = next(((x, y) for x, y, n in pts if n == "origin"), (None, None))
    if ox is not None:
        ax.axvline(ox, color="#FFD700", lw=1.0, ls=":", alpha=0.7, zorder=0)
        ax.axhline(oy, color="#FFD700", lw=1.0, ls=":", alpha=0.7, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.margins(0.14)
    ax.grid(alpha=0.25)
    return handles, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="../product/results/_analysis")
    ap.add_argument("--outdir", default="../references/think/figs")
    ap.add_argument("--xpad", type=float, default=6.0)
    ap.add_argument("--ypad", type=float, default=3.0)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()
    root, out = Path(args.eval_root), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    D = {t: load(root, t) for t in MODELS}
    D = {t: v for t, v in D.items() if v}
    tags = list(D)

    print(f"{'model':<14} | " + " | ".join(f"{s}-{k} a/d/oe" for s in ("for", "ret", "rea")
                                            for k in TASKS))
    for t in tags:
        v = D[t]
        print(f"{t:<14} | " + " | ".join(
            f"{100*v[(s,k)]['a']:4.1f}/{100*v[(s,k)]['d']:4.1f}/{100*v[(s,k)]['oe']:4.1f}"
            for s in ("forget", "retain", "real") for k in TASKS))

    # ---- Class 1: 3x2 iso-oe grid -------------------------------------------
    rows = [("forget", "better: a up / oe down"),
            ("retain", "better: d up / oe down"),
            ("real", "better: d up / oe down")]
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 15))
    h = l = None
    for r, (scope, good) in enumerate(rows):
        for c, task in enumerate(TASKS):
            pts = [(100 * D[t][(scope, task)]["d"], 100 * D[t][(scope, task)]["a"], t)
                   for t in tags]
            h, l = draw_ad(axes[r, c], pts, scope, task, good,
                           args.xpad, args.ypad, args.step)
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("Consistency / efficiency: a-d plane with iso-oe lines (a+d+oe=100).  "
                 "At same d, higher a = lower oe.\n"
                 "forget = forget set; retain = shared; real = real person.  "
                 "gold star = origin; reds = clip max_norm; grey = no-clip grad_scale",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.965))
    save(fig, out, "fig_p11_isooe_grid")
    plt.close(fig)

    # ---- Class 2: 2x2 Pareto grid -------------------------------------------
    cols = [("retain", "retain(shared)-d"), ("real", "real-d")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    h = l = None
    for r, task in enumerate(TASKS):
        for c, (scope, ylab) in enumerate(cols):
            pts = [(100 * D[t][("forget", task)]["a"], 100 * D[t][(scope, task)]["d"], t)
                   for t in tags]
            h, l = draw_pareto(axes[r, c], pts,
                               f"forget-{task} a = AllErr (higher = both forgotten)",
                               f"{ylab} (higher = kept)",
                               f"{task}: forget-a  x  {scope}-d")
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("forget-retain / forget-real Pareto (upper-right = Pareto; "
                 "gold star = origin, gold dotted = origin reference)\n"
                 "higher forget-a is better; higher retain/real-d is better",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save(fig, out, "fig_p12_pareto_grid")
    plt.close(fig)

    print("saved fig_p11_isooe_grid, fig_p12_pareto_grid to", out)


if __name__ == "__main__":
    main()
