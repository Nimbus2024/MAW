#!/usr/bin/env python3
"""P1/P2 consistency visualization for the MAW ablation.

Reuses the rendering of the reference `plot_consistency.py`
(draw_ad / draw_pareto from plot_consistency, colors/save from plot_pareto),
only swapping the data source for the ablation runs:

    a = AllErr = P(both wrong);  d = AllAcc = P(both right);  oe = 1-a-d
    forget = Forget Set;  retain = Retain Set (shared);  real = Retain Set (real person)

Run with the matplotlib env:  /home/jwzm/miniconda3/envs/d2l/bin/python
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_pareto as PP  # noqa: E402  (save)
import plot_consistency as PC  # noqa: E402  (draw_ad, draw_pareto, TASKS, SETS)

SETS = {"forget": "Forget Set Results",
        "retain": "Retain Set (shared dataset) Results",
        "real": "Retain Set (real person) Results"}
TASKS = ("fill", "cls")

# tag, glob (under <root>/results), label, color, epoch (None=final), star?
RUNS = [
    ("origin", "ratio5/origin/20260906_204146", "origin", "#FFD700", 0, True),
    ("oldMAW", "ratio5/MAW/20260907_170751", "oldMAW 170751 (l=1.5)", "#8B0000", 6, False),
    ("phaseC", "ablation/MAW/*-emaRL_a0.3", "PhaseC a=0.3 (l=1.5)", "#E85A5A", 6, False),
    ("v1ema", "ablation/MAW/*-v1ema", "v1ema (dyn EMA, l=0)", "#1f77b4", 4, False),
    ("v4it", "ablation/MAW/*-v4it", "v4it (IT-only, l=0)", "#2ca02c", 4, False),
    ("v5pt", "ablation/MAW/*-v5pt", "v5pt (PT-only, l=0)", "#ff7f0e", 4, False),
    ("v6a3", "ablation/MAW/*-v6a3", "v6a3 (alpha=3, l=0)", "#9467bd", 4, False),
    ("v8b04", "ablation/MAW/*-v8b04", "v8b04 (beta=.4, l=0)", "#8c564b", 4, False),
    ("v9coeff1", "ablation/MAW/*-v9coeff1", "v9coeff1 (coeff=1, l=0)", "#e377c2", 4, False),
    ("v10dpo", "ablation/DPO/*-v10dpo", "v10dpo (DPO, l=0)", "#17becf", 4, False),
    ("v11npo", "ablation/NPO/*-v11npo", "v11npo (NPO, l=0)", "#bcbd22", 4, False),
]


def _final_json(run_dir, epoch):
    if epoch:
        c = sorted(glob.glob(f"{run_dir}/runs/epoch-{epoch}/metrics/*final_evaluation_results.json"))
        if c:
            return c[-1]
    c = (sorted(glob.glob(f"{run_dir}/runs/epoch-*/metrics/*final_evaluation_results.json"))
         or sorted(glob.glob(f"{run_dir}/metrics/*final_evaluation_results.json")))
    return c[-1] if c else None


def load(root, tag):
    entry = next(r for r in RUNS if r[0] == tag)
    dirs = sorted(glob.glob(str(Path(root) / "results" / entry[1])))
    if not dirs:
        return None
    fj = _final_json(dirs[0], entry[4])
    if not fj:
        return None
    m = json.load(open(fj))
    out = {}
    for scope, skey in SETS.items():
        for task in TASKS:
            tkey = "fill_in_the_blank" if task == "fill" else "classification"
            b = m[skey][tkey]
            d = b["All Modal Question Accuracy"] / 100
            a = b["All Modal Question Error"] / 100
            out[(scope, task)] = {"a": a, "d": d, "oe": 1 - a - d}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../product")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--xpad", type=float, default=6.0)
    ap.add_argument("--ypad", type=float, default=3.0)
    ap.add_argument("--step", type=int, default=10)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.outdir) if args.outdir else root / "record" / "ablations" / "groups"
    out.mkdir(parents=True, exist_ok=True)

    # feed the ablation palette/labels into the reference draw functions
    PC.COLORS = {r[0]: r[3] for r in RUNS}
    PC.LABELS = {r[0]: r[2] for r in RUNS}
    PP.COLORS = PC.COLORS
    PP.LABELS = PC.LABELS

    D = {r[0]: load(root, r[0]) for r in RUNS}
    D = {t: v for t, v in D.items() if v}
    tags = list(D)
    print("models:", tags)
    for t in tags:
        v = D[t]
        print(f"{t:10s} | " + " | ".join(
            f"{s}-{k} {100*v[(s,k)]['a']:.0f}/{100*v[(s,k)]['d']:.0f}/{100*v[(s,k)]['oe']:.0f}"
            for s in ("forget", "retain", "real") for k in TASKS))

    # ---- fig_p11: 3x2 iso-oe grid -------------------------------------------
    rows = [("forget", "better: a up / oe down"),
            ("retain", "better: d up / oe down"),
            ("real", "better: d up / oe down")]
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 15))
    h = l = None
    for r, (scope, good) in enumerate(rows):
        for c, task in enumerate(TASKS):
            pts = [(100 * D[t][(scope, task)]["d"], 100 * D[t][(scope, task)]["a"], t) for t in tags]
            h, l = PC.draw_ad(axes[r, c], pts, scope, task, good, args.xpad, args.ypad, args.step)
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("MAW ablation consistency/efficiency: a-d plane with iso-oe lines (a+d+oe=100). "
                 "At same d, higher a = lower oe.\n"
                 "forget = forget set; retain = shared; real = real person.  "
                 "gold star = origin; reds = lambda=1.5 controls; others = lambda=0 families",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    PP.save(fig, out, "fig_p11_isooe_ablation")
    plt.close(fig)

    # ---- fig_p12: 2x2 Pareto grid -------------------------------------------
    cols = [("retain", "retain(shared)-d"), ("real", "real-d")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    h = l = None
    for r, task in enumerate(TASKS):
        for c, (scope, ylab) in enumerate(cols):
            pts = [(100 * D[t][("forget", task)]["a"], 100 * D[t][(scope, task)]["d"], t) for t in tags]
            h, l = PC.draw_pareto(axes[r, c], pts,
                                  f"forget-{task} a = AllErr (higher = both forgotten)",
                                  f"{ylab} (higher = kept)",
                                  f"{task}: forget-a  x  {scope}-d")
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("MAW ablation forget-retain / forget-real Pareto (upper-right = Pareto; "
                 "gold star = origin, gold dotted = origin reference)\n"
                 "higher forget-a is better; higher retain/real-d is better",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    PP.save(fig, out, "fig_p12_pareto_ablation")
    plt.close(fig)

    print("saved to", out)


if __name__ == "__main__":
    main()
