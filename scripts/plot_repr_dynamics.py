#!/usr/bin/env python3
"""表示 / 梯度 的跨模态相似度折线图.

产物:
  fig_repr_cka_layers    源 cka_layers.json  -> 逐层 CKA(IT, PT)
  fig_grad_cos_steps     源 p02_p0clip1.0_*/logs/grad_norms.jsonl -> 逐 step cos(g_mm, g_um)
                         (左: 全局曲线 + 64 组 mean±std 带; 右: attn/mlp 两张 层×step 热力图)
  fig_update_cos_layers  源 e2_geometry.json -> 逐层 cos(ΔW_mm, ΔW_um)
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_COLORS = {"base": "#B8860B", "mm": "#1f77b4", "um": "#ff7f0e", "joint": "#9467bd"}
MODELS = ("base", "mm", "um", "joint")
LAYERS_64 = [(L, k) for L in range(32) for k in ("attn", "mlp")]


def find(root, pat):
    hits = sorted(glob.glob(str(Path(root) / pat), recursive=True))
    if not hits:
        raise FileNotFoundError(pat)
    return hits[0]


def save(fig, outdir, name):
    fig.savefig(Path(outdir) / f"{name}.svg")
    fig.savefig(Path(outdir) / f"{name}.png", dpi=140)


def fig_cka_layers(root, out):
    d = json.load(open(find(root, "cka_layers.json")))
    layers, n = d["layers"], d["n"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for metric, ax, ylab in (("cka_it_pt", axes[0], "CKA(IT, PT)"),
                             ("cos_it_pt", axes[1], r"$\cos$(IT, PT)")):
        for m in MODELS:
            y = [d["models"][m][str(L)][metric] for L in layers]
            ax.plot(layers, y, "o-", color=MODEL_COLORS[m], lw=1.6, ms=3.2, label=m)
        ax.set_xlabel("layer index")
        ax.set_ylabel(ylab)
        ax.set_xticks(range(0, len(layers), 4))
        ax.set_xlim(-1, len(layers))
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_title(f"linear-kernel CKA  (n={n}, forget_5)")
    axes[1].set_title("mean-pooled representation cosine")
    fig.suptitle("IT vs PT representations per layer  "
                 "(0 = embeddings, 1..32 = LM layers; low = near-orthogonal)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out, "fig_repr_cka_layers")
    plt.close(fig)


def _grad_matrix(rows):
    n = len(rows)
    attn = np.full((32, n), np.nan)
    mlp = np.full((32, n), np.nan)
    for j, r in enumerate(rows):
        for L in range(32):
            attn[L, j] = r["grad"][f"layer_{L}_attn"]["cos"]
            mlp[L, j] = r["grad"][f"layer_{L}_mlp"]["cos"]
    return attn, mlp


def fig_grad_cos_steps(root, out, vm=0.25):
    rows = [json.loads(l) for l in open(find(root, "**/p02_p0clip1.0_*/logs/grad_norms.jsonl"))]
    steps = np.array([r["step"] for r in rows])
    gcos = np.array([r["grad"]["_global"]["cos"] for r in rows])
    G = np.array([[r["grad"][f"layer_{L}_{k}"]["cos"] for L, k in LAYERS_64] for r in rows])
    mean, std = G.mean(1), G.std(1)
    attn, mlp = _grad_matrix(rows)

    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], wspace=0.30, hspace=0.14)
    axL = fig.add_subplot(gs[:, 0])
    axT = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, 1], sharex=axT, sharey=axT)

    axL.plot(steps, gcos, color="black", lw=1.6, label=r"global  $\cos(g_{mm},g_{um})$")
    axL.fill_between(steps, mean - std, mean + std, color="#1f77b4", alpha=0.25,
                     label="64 groups: mean ± std")
    axL.axhline(0, color="gray", ls="--", lw=0.8)
    axL.set_xlabel("training step")
    axL.set_ylabel("cosine")
    axL.set_title(f"per-step cross-modal gradient cosine  (mean={gcos.mean():.3f})")
    axL.legend(frameon=False, fontsize=8)
    axL.grid(alpha=0.3)

    for ax, M, name in ((axT, attn, "attn"), (axB, mlp, "mlp")):
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="coolwarm",
                       vmin=-vm, vmax=vm,
                       extent=[steps[0] - 0.5, steps[-1] + 0.5, -0.5, 31.5])
        ax.set_ylabel(f"layer ({name})")
        ax.set_yticks([0, 8, 16, 24, 31])
    axT.tick_params(labelbottom=False)
    axT.set_title(r"per-layer $\cos(g_{mm},g_{um})$ over steps")
    axB.set_xlabel("training step")
    axB.set_xticks([0, 20, 40, 60, 80, 100])
    fig.colorbar(im, ax=[axT, axB], location="right", shrink=0.85, label="cosine")
    fig.suptitle("cross-modal gradient cosine during NPO joint training  "
                 "(near 0 = orthogonal)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "fig_grad_cos_steps")
    plt.close(fig)


def fig_update_cos_layers(root, out):
    d = json.load(open(find(root, "e2_geometry.json")))
    per = d["comparison"]["per_layer"]
    ls = sorted(int(k) for k in per)
    y = [per[str(L)]["cos"] for L in ls]
    gc = d["comparison"]["global"]["cos"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(ls, y, "o-", color="#2ca02c", lw=1.8, ms=5, label="per-layer")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axhline(gc, color="black", ls=":", lw=1.2, label=f"global = {gc:.3f}")
    ax.set_xticks(list(range(0, 32, 2)))
    ax.set_xlabel("layer index")
    ax.set_ylabel(r"$\cos(\Delta W_{mm}, \Delta W_{um})$")
    ax.set_title("per-layer update (LoRA delta) cosine between modalities")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, out, "fig_update_cos_layers")
    plt.close(fig)


def fig_c1_baseline(root, out):
    pools = [("mean", "#1f77b4"), ("last", "#ff7f0e"),
             ("answer", "#2ca02c"), ("prompt", "#d62728")]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    for p, col in pools:
        d = json.load(open(find(root, f"cka_layers_{p}.json")))
        L = d["layers"]
        b = d["models"]["base"]
        axes[0].plot(L, [b[str(l)]["cos_it_pt"] for l in L], "-", color=col, lw=1.8, label=f"{p}")
        axes[0].plot(L, [b[str(l)]["cos_it_pt_x"] for l in L], "--", color=col, lw=1.1)
        axes[1].plot(L, [b[str(l)]["cos_it_it"] for l in L], "-", color=col, lw=1.8, label=f"{p}")
        axes[1].plot(L, [b[str(l)]["cos_pt_pt"] for l in L], "--", color=col, lw=1.1)
    axes[0].set_title(r"cross-modal $\cos$(IT,PT):  solid = same question, dashed = different question")
    axes[0].set_ylabel(r"$\cos$(IT,PT)")
    axes[1].set_title(r"same-modality, different questions:  solid = IT-IT, dashed = PT-PT")
    for ax in axes:
        ax.set_xlabel("layer index")
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("C1: is the cross-modal similarity question-specific?  "
                 "(solid vs dashed overlapping = it is NOT)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out, "fig_c1_baseline")
    plt.close(fig)


def fig_c4b(root, out):
    scopes = ["forget", "retain_shared", "retain_celebrity"]
    gcol = {"TT": "#2ca02c", "one": "#ff7f0e", "FF": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, sc in zip(axes, scopes):
        d = json.load(open(find(root, f"qa_shared_{sc}.json")))
        L = d["layers"]
        for lab in ("TT", "one", "FF"):
            y = [d["groups"][lab][str(l)]["cos"] for l in L]
            ax.plot(L, y, "o-", color=gcol[lab], ms=3, lw=1.6,
                    label=f"{lab} (n={d['counts'][lab]})")
        ax.set_title(sc)
        ax.set_xlabel("layer index")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(r"cross-modal $\cos$(IT,PT)  (mean pool)")
    fig.suptitle("C4-b: shared component by base-model correctness  "
                 "(TT = both right, one = one-sided, FF = both wrong)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, "fig_c4b_shared_groups")
    plt.close(fig)


def fig_c3_ablation(root, out):
    d = json.load(open(find(root, "shared_ablation.json")))
    scopes = ["forget", "retain_shared", "retain_celebrity"]
    layers = sorted(d["layers"], key=int)
    fig, axes = plt.subplots(1, len(layers), figsize=(6.5 * len(layers), 4.4))
    if len(layers) == 1:
        axes = [axes]
    for ax, L in zip(axes, layers):
        e = d["layers"][L]["scopes"]
        x = np.arange(len(scopes))
        w = 0.38
        ax.bar(x - w / 2, [e[sc]["Image_Textual"]["delta"] for sc in scopes], w,
               color="#1f77b4", label="IT")
        ax.bar(x + w / 2, [e[sc]["Pure_Text"]["delta"] for sc in scopes], w,
               color="#ff7f0e", label="PT")
        ax.set_xticks(x)
        ax.set_xticklabels(scopes, fontsize=8)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"layer {L}")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(r"$\Delta$CE after removing shared subspace")
    fig.suptitle(f"C3: project out top-k cross-modal shared directions (k={d['k']})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, "fig_c3_shared_ablation")
    plt.close(fig)


def fig_c1_segments(root, out):
    pools = ["mean", "noimg", "image"]
    colors = {"mean": "#555555", "noimg": "#1f77b4", "image": "#d62728"}
    D = {}
    for p in pools:
        try:
            D[p] = json.load(open(find(root, f"cka_layers_{p}.json")))
        except FileNotFoundError:
            print(f"[fig_c1_segments] missing cka_layers_{p}.json -> skipped")
            return
    L = D["mean"]["layers"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for p in pools:
        b = D[p]["models"]["base"]
        axes[1].plot(L, [b[str(l)]["cos_it_it"] for l in L], "-o", ms=3,
                     color=colors[p], lw=1.6, label=p)
    for p in ("mean", "noimg"):
        b = D[p]["models"]["base"]
        axes[0].plot(L, [b[str(l)]["cos_it_pt"] for l in L], "-o", ms=3,
                     color=colors[p], lw=1.6, label=f"{p} (same-q)")
        axes[0].plot(L, [b[str(l)]["cos_it_pt_x"] for l in L], "--", ms=3,
                     color=colors[p], lw=1.1, label=f"{p} (diff-q)")
    bim = D["image"]["models"]["base"]
    if "cos_it_img_noimg" in bim[str(L[0])]:
        axes[0].plot(L, [bim[str(l)]["cos_it_img_noimg"] for l in L], "-.", ms=3,
                     color="#2ca02c", lw=1.6, label="IT image vs IT text")
    for p in ("mean", "noimg"):
        b = D[p]["models"]["base"]
        axes[2].plot(L, [b[str(l)]["cos_pt_pt"] for l in L], "-o", ms=3,
                     color=colors[p], lw=1.6, label=p)
    axes[0].set_title(r"cross-modal  $\cos$(IT,PT)")
    axes[1].set_title(r"same-modality IT-IT, diff questions")
    axes[2].set_title(r"same-modality PT-PT, diff questions")
    axes[1].set_ylabel("cosine")
    for ax in axes:
        ax.set_xlabel("layer index")
        ax.set_ylim(0, 1.02)
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("C1 segments: image-token vs non-image-token pooling  "
                 "(image pool only on IT; PT has no image tokens)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, "fig_c1_segments")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="../product/results/_analysis")
    ap.add_argument("--outdir", default="../references/think/figs")
    ap.add_argument("--vmax", type=float, default=0.25)
    args = ap.parse_args()
    root, out = Path(args.eval_root), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    fig_cka_layers(root, out)
    fig_grad_cos_steps(root, out, args.vmax)
    fig_update_cos_layers(root, out)
    fig_c1_baseline(root, out)
    fig_c1_segments(root, out)
    fig_c4b(root, out)
    fig_c3_ablation(root, out)
    print("saved fig_repr_cka_layers, fig_grad_cos_steps, fig_update_cos_layers, "
          "fig_c1_baseline, fig_c1_segments, fig_c4b_shared_groups, fig_c3_shared_ablation to", out)


if __name__ == "__main__":
    main()
