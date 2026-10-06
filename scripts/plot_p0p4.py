#!/usr/bin/env python3
"""P0-P4 可视化: 从 results/_analysis 读取数据, 输出 SVG 到 references/think/figs/。

图:
  fig_p0_grad_scales.svg   逐模态梯度尺度全轨迹 (+ clip 阈值)
  fig_p1_balance.svg       逐模态归一前后: ΔW 方向对齐 + 行为效应占比
  fig_p2_clip.svg          clip 阈值扫描 + 无 clip/grad_scale 对照 (Adam 尺度不变)
  fig_p3_additivity.svg    base/mm/um/mm+um/joint 的行为 (叠加性)
  fig_p4_dose.svg          单模态剂量曲线 (局部线性)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEARCH_DIRS = []


def find(pattern):
    hits = []
    for d in SEARCH_DIRS:
        hits += sorted(glob.glob(str(Path(d) / pattern)))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[-1]


def save(fig, outdir, name):
    fig.savefig(Path(outdir) / f"{name}.svg")
    fig.savefig(Path(outdir) / f"{name}.png", dpi=140)


def ce(path):
    o = json.load(open(path))["p1_qa"]["overall"]
    return o["it"]["mean"], o["pt"]["mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="../product/results/_analysis")
    ap.add_argument("--eval-root", default="../product/results/_analysis")
    ap.add_argument("--outdir", default="../references/think/figs")
    args = ap.parse_args()
    global SEARCH_DIRS
    A = Path(args.analysis)
    E = Path(args.eval_root)
    O = Path(args.outdir)
    SEARCH_DIRS = [A, A / "p02_full"]
    O.mkdir(parents=True, exist_ok=True)

    base_it, base_pt = ce(find("ce_breakdown.json"))
    mm_it, _ = ce(find("p02_refmm_*.ce.json"))
    _, um_pt = ce(find("p02_refum_*.ce.json"))

    def frac(tag):
        it, pt = ce(find(f"p02_{tag}*.ce.json"))
        return (it - base_it) / (mm_it - base_it), (pt - base_pt) / (um_pt - base_pt), it, pt

    dj = json.load(open(find("dw_proj_p0p1.json")))
    dc = json.load(open(find("dw_proj_clips.json")))
    dp = json.load(open(find("dw_proj_p2c.json")))

    # ---------------- Fig 0: P0 grad scales ----------------
    rows = [json.loads(l) for l in open(find("p02_p0clip1.0_*/logs/grad_norms.jsonl"))]
    steps = np.array([r["step"] for r in rows])

    def tot(r, k):
        return math.sqrt(sum(v[k] ** 2 for v in r["grad"].values() if isinstance(v, dict) and k in v))

    g_mm = np.array([tot(r, "norm_mm") for r in rows])
    g_um = np.array([tot(r, "norm_um") for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogy(steps, g_um, label=r"$\|g_{um}\|$ (PT)", color="tab:red")
    ax[0].semilogy(steps, g_mm, label=r"$\|g_{mm}\|$ (IT)", color="tab:blue")
    ax[0].axhline(1.0, ls="--", color="gray", label="clip threshold = 1.0")
    ax[0].set_xlabel("training step"); ax[0].set_ylabel(r"gradient norm (log)")
    ax[0].set_title("P0: per-modality gradient norms (joint training)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")
    ax[1].semilogy(steps, g_um / g_mm, color="tab:purple")
    ax[1].set_xlabel("training step"); ax[1].set_ylabel(r"$\|g_{um}\|/\|g_{mm}\|$ (log)")
    ax[1].set_title("scale ratio: 3-257x throughout")
    ax[1].grid(alpha=.3, which="both")
    fig.tight_layout(); save(fig, O, "fig_p0_grad_scales"); plt.close(fig)

    # ---------------- Fig 1: P1 balance ----------------
    p0f = frac("p0clip1.0"); p1f = frac("p1bal")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(2); w = 0.35
    c0 = dj["targets"]["p0clip1.0"]; c1 = dj["targets"]["p1bal"]
    ax[0].bar(x - w / 2, [c0["cos_mm"], c0["cos_um"]], w, label="joint (clip=1)", color="tab:orange")
    ax[0].bar(x + w / 2, [c1["cos_mm"], c1["cos_um"]], w, label="joint + grad-balance", color="tab:green")
    ax[0].set_xticks(x); ax[0].set_xticklabels([r"$\cos(\Delta W,\Delta W_{mm})$", r"$\cos(\Delta W,\Delta W_{um})$"])
    ax[0].set_ylabel("cosine"); ax[0].set_title(r"$\Delta W$ direction (um-dominated $\to$ balanced)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
    ax[1].bar(x - w / 2, [p0f[0], p0f[1]], w, label="joint (clip=1)", color="tab:orange")
    ax[1].bar(x + w / 2, [p1f[0], p1f[1]], w, label="joint + grad-balance", color="tab:green")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["mm effect\n(IT, % of mm-only)", "um effect\n(PT, % of um-only)"])
    ax[1].set_ylabel("fraction of single-modality effect")
    ax[1].set_title("behavioral effect realization")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); save(fig, O, "fig_p1_balance"); plt.close(fig)

    # ---------------- Fig 2: P2 clip / Adam ----------------
    clips = [("0.2", "p02_clip0.2"), ("1.0", "p02_p0clip1.0"), ("5.0", "p02_clip5.0"), ("100", "p02_clip100.0")]
    xs = [0.2, 1.0, 5.0, 100.0]
    its = [ce(find(f"{p}*.ce.json"))[0] for _, p in clips]
    pts = [ce(find(f"{p}*.ce.json"))[1] for _, p in clips]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].semilogx(xs, its, "o-", color="tab:blue", label="IT (mm)")
    ax[0].semilogx(xs, pts, "s-", color="tab:red", label="PT (um)")
    ax[0].axhline(base_it, ls=":", color="tab:blue", alpha=.6); ax[0].axhline(base_pt, ls=":", color="tab:red", alpha=.6)
    ax[0].axhline(mm_it, ls="--", color="tab:blue", alpha=.6); ax[0].axhline(um_pt, ls="--", color="tab:red", alpha=.6)
    ax[0].set_xlabel("clip max_norm"); ax[0].set_ylabel("CE (question-avg)")
    ax[0].set_title("P2: clip value 0.2/1/5 identical; no-clip learns less")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")
    labels = ["clip=0.2", "clip=1", "clip=5", "noclip", "noclip\n+gs0.04", "noclip\n+gs0.20", "grad\nbalance"]
    cums = [dc["targets"]["clip0.2"]["cos_um"], dc["targets"]["clip5"]["cos_um"],
            dp["targets"]["noclip_gs0.04"]["cos_um"], dp["targets"]["noclip_gs0.20"]["cos_um"],
            c1["cos_um"]]
    vals = [dc["targets"]["clip0.2"]["cos_um"], c0["cos_um"], dc["targets"]["clip5"]["cos_um"],
            dc["targets"]["clip100"]["cos_um"], dp["targets"]["noclip_gs0.04"]["cos_um"],
            dp["targets"]["noclip_gs0.20"]["cos_um"], c1["cos_um"]]
    cols = ["tab:orange"] * 3 + ["tab:gray"] * 3 + ["tab:green"]
    ax[1].bar(range(len(vals)), vals, color=cols)
    ax[1].set_xticks(range(len(vals))); ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_ylabel(r"$\cos(\Delta W,\Delta W_{um})$"); ax[1].set_ylim(0, 1)
    ax[1].set_title("Adam is scale-invariant (gs$\\approx$noclip); clip$\\neq$cause")
    ax[1].grid(alpha=.3, axis="y")
    fig.tight_layout(); save(fig, O, "fig_p2_clip"); plt.close(fig)

    # ---------------- Fig 3: P3 additivity ----------------
    d = json.load(open(find("dw_dose.json")))
    C = d["configs"]; J = d["joint"]
    keys = [("base", "0,0"), ("mm-only", "1,0"), ("um-only", "0,1"), ("mm+um (added)", "1,1"), ("joint (trained)", None)]
    its = [C[k]["it"]["mean"] if k else J["it"]["mean"] for _, k in keys]
    pts = [C[k]["pt"]["mean"] if k else J["pt"]["mean"] for _, k in keys]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(keys)); w = 0.38
    ax.bar(x - w / 2, its, w, label="IT (mm)", color="tab:blue")
    ax.bar(x + w / 2, pts, w, label="PT (um)", color="tab:red")
    for i, (a, b) in enumerate(zip(its, pts)):
        ax.text(i - w / 2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in keys])
    ax.set_ylabel("CE (question-avg)")
    ax.set_title("P3: ADDING the two updates keeps BOTH effects; joint TRAINING loses mm")
    ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); save(fig, O, "fig_p3_additivity"); plt.close(fig)

    # ---------------- Fig 4: P4 dose curves ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    lam_mm = [0.0, 0.5, 1.0, 1.5]
    it_mm = [C[f"{a:g},0"]["it"]["mean"] for a in lam_mm]
    ax[0].plot(lam_mm, it_mm, "o-", color="tab:blue", label="measured")
    ax[0].plot([0, 1.5], [it_mm[0], it_mm[0] + 1.5 * (it_mm[2] - it_mm[0]) / 1.0], "--",
               color="gray", label="linear from $\\lambda$=0,1")
    ax[0].set_xlabel(r"$\lambda$  (scale on $\Delta W_{mm}$)"); ax[0].set_ylabel("IT CE")
    ax[0].set_title("P4: mm dose (convex/super-linear)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    lam_um = [0.0, 0.5, 1.0, 1.5]
    pt_um = [C[f"0,{b:g}"]["pt"]["mean"] for b in lam_um]
    ax[1].plot(lam_um, pt_um, "s-", color="tab:red", label="measured")
    ax[1].set_xlabel(r"$\beta$  (scale on $\Delta W_{um}$)"); ax[1].set_ylabel("PT CE")
    ax[1].set_title("P4: um dose")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout(); save(fig, O, "fig_p4_dose"); plt.close(fig)

    import glob as _glob

    def raw(tag):
        f = _glob.glob(str(E / f"p02_eval_{tag}" / f"{tag}_final_evaluation_results.json"))
        if not f:
            return None
        d = json.load(open(f[0]))
        o = {}
        for key, scope in (("forget", "Forget Set Results"),
                           ("retain", "Retain Set (shared dataset) Results"),
                           ("real", "Retain Set (real person) Results")):
            s = d[scope]
            o[key] = {
                "fill": (s["fill_in_the_blank"]["image_textual_accuracy"], s["fill_in_the_blank"]["pure_text_accuracy"]),
                "cls": (s["classification"]["Image-Textual Question Accuracy"], s["classification"]["Pure Text Question Accuracy"]),
                "gen": (s["generation"]["Average ROUGE-L (Image_Textual)"], s["generation"]["Average ROUGE-L (Pure_Text)"]),
            }
        return o

    panels = [("forget", "fill", "Forget Fill  (lower=better)"), ("forget", "cls", "Forget Classif  (lower=better)"),
              ("forget", "gen", "Forget Gen ROUGE-L  (lower=better)"), ("retain", "fill", "Retain Fill  (higher=better)"),
              ("retain", "cls", "Retain Classif  (higher=better)"), ("real", "cls", "Real Classif  (higher=better)")]

    def raw_fig(models, fname, suptitle):
        models = [t for t in models if raw(t)]
        if not models:
            return
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for ax, (scope, task, title) in zip(axes.ravel(), panels):
            it = [raw(t)[scope][task][0] for t in models]
            pt = [raw(t)[scope][task][1] for t in models]
            x = np.arange(len(models)); w = 0.38
            ax.bar(x - w / 2, it, w, label="IT (mm)", color="tab:blue")
            ax.bar(x + w / 2, pt, w, label="PT (um)", color="tab:red")
            ax.set_xticks(x); ax.set_xticklabels(models, rotation=40, ha="right", fontsize=7)
            ax.set_title(title, fontsize=10); ax.grid(alpha=.3, axis="y")
            if task in ("fill", "cls"):
                ax.set_ylim(0, 105)
        axes[0, 0].legend(fontsize=9)
        fig.suptitle(suptitle, fontsize=12)
        fig.tight_layout(); save(fig, O, fname)

    raw_fig(["origin", "refmm", "refum", "p0clip1.0", "p1bal", "merged_mm_um"],
            "fig_p6_eval_raw", "UMU-bench raw metrics (IT vs PT): mechanism group")
    raw_fig(["origin", "p0clip1.0", "clip0.2", "clip5.0", "clip100.0",
             "noclip_gs0.04", "noclip_gs0.20", "noclip_lr5x"],
            "fig_p7_eval_raw_p2", "UMU-bench raw metrics (IT vs PT): P2 clip/scale controls")

    print("saved figures to", O)


if __name__ == "__main__":
    main()
