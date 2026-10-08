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
    print("saved fig_repr_cka_layers, fig_grad_cos_steps, fig_update_cos_layers to", out)


if __name__ == "__main__":
    main()
