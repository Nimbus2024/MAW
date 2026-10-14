#!/usr/bin/env python3
"""可视化 grad_subspace 的能量图: C = U_K^T gW_thr V_K 的 |C|^2（W 的 SVD 基下）。

fig_gradsub_emaps : 行=层(8/16/24/31)，列=[q_proj·mm, q_proj·um, down_proj·mm, down_proj·um]
                    每格 log10|C|^2 热力图（对角线为白虚线），标题给出对角能量占比。
fig_gradsub_diag  : 对角剖面 |diag(C)|^2 vs 奇异方向序号 j（mm/um 各一条，若干层）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUFFIX = ""


def save(fig, outdir, name):
    name = name + SUFFIX
    fig.savefig(Path(outdir) / f"{name}.svg")
    fig.savefig(Path(outdir) / f"{name}.png", dpi=150)


PROJS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def _topq(E, q=1.0):
    n = E.size
    k = max(1, int(round(q / 100.0 * n)))
    thr = np.partition(E.ravel(), -k)[-k]
    return E.ravel() >= thr


def _jaccard(a, b):
    i = np.logical_and(a, b).sum()
    u = np.logical_or(a, b).sum()
    return float(i) / float(u) if u else float("nan")


def _spearman(x, y):
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / d) if d > 0 else float("nan")


def _pair_colors(n):
    base = list(plt.get_cmap("tab10").colors)
    return [base[i % len(base)] for i in range(n)]


def _baseline(pl, layers, ckey, n_pair):
    """同模态异题基线：对 pair 1..n-1 取均值（单 pair 噪声大）。"""
    ys = []
    for ti in range(1, n_pair):
        col = [pl[str(L)].get(str(ti), {}).get(ckey) for L in layers]
        if any(v is not None for v in col):
            ys.append(col)
    if not ys:
        return []
    return [np.nanmean([y[i] for y in ys]) if any(y[i] is not None for y in ys) else np.nan
            for i in range(len(layers))]


def fig_A(jsonpath, outdir, q=1.0):
    """A 方案：完整能量图 |C|^2 的 support(Jaccard) 与能量图相关(Spearman)，3 个 pair 逐层。"""
    import json
    d = json.load(open(jsonpath))
    pl = d["per_layer"]
    layers = sorted(int(k) for k in pl)
    n_pair = len(d["per_target"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, key, ckey, ttl, yl, rnd in (
            (axes[0], "jaccard_mm_um", "jaccard_cross_it",
             "support overlap (top-%g%% Jaccard)" % q, "Jaccard", q / 100.0),
            (axes[1], "spearman_mm_um", "spearman_cross_it",
             "energy-map correlation (Spearman)", r"$\rho$", 0.0)):
        for ti, col in enumerate(_pair_colors(n_pair)):
            y = [pl[str(L)].get(str(ti), {}).get(key) for L in layers]
            ax.plot(layers, y, "-o", ms=2.6, lw=1.4, color=col, label=f"mm vs um (pair {ti})")
        yc = _baseline(pl, layers, ckey, n_pair)
        if any(v is not None for v in yc):
            ax.plot(layers, yc, "--", color="black", lw=1.3, label="IT$_i$ vs IT$_0$ (diff. pair)")
        ax.axhline(rnd, ls=":", color="gray", lw=1.2, label=f"random (={rnd:g})")
        ax.set_xlabel("layer index")
        ax.set_ylabel(yl)
        ax.set_title("A: " + ttl)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("A: full |C|$^2$ energy-map overlap in W's SVD basis  "
                 "(all %d target pairs)" % n_pair, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, outdir, "fig_gradsub_A_emaps")
    plt.close(fig)


def fig_B(jsonpath, outdir):
    """B 方案：gW 自身 top-r 子空间重叠 vs 层深度（3 个目标对 + 基线）。"""
    import json
    d = json.load(open(jsonpath))
    pl = d["per_layer"]
    layers = sorted(int(k) for k in pl)
    n_pair = len(d["per_target"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, key, rnd_key, ttl in ((axes[0], "subL_mm_um", "subL_rand", "left subspace overlap"),
                                  (axes[1], "subR_mm_um", "subR_rand", "right subspace overlap")):
        for ti, col in enumerate(_pair_colors(n_pair)):
            y = [pl[str(L)].get(str(ti), {}).get(key) for L in layers]
            ax.plot(layers, y, "-o", ms=2.6, lw=1.5, color=col, label=f"mm vs um (pair {ti})")
        ckey = "subL_it_i0" if key == "subL_mm_um" else "subR_it_i0"
        yc = _baseline(pl, layers, ckey, n_pair)
        if all(v is not None for v in yc):
            ax.plot(layers, yc, "--", color="black", lw=1.3, label="IT$_i$ vs IT$_0$ (diff. pair)")
        yr = [pl[str(L)].get("0", {}).get(rnd_key) for L in layers]
        if all(v is not None for v in yr):
            ax.plot(layers, yr, ":", color="gray", lw=1.3, label="random subspace")
        ax.set_xlabel("layer index")
        ax.set_ylabel("overlap fraction")
        ax.set_title("B: " + ttl)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("B: overlap of the gradient's OWN top-16 subspaces "
                 "(mm/um largely disjoint; small above-random, question-specific part)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, outdir, "fig_gradsub_B_layers")
    plt.close(fig)


def fig_proj(jsonpath, outdir):
    """逐投影(7 个)分解：B 的 subL/subR 与 A 的 Jaccard/Spearman（3 个 pair + 同模态基线）。"""
    import json
    d = json.load(open(jsonpath))
    projs = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    panels = [("subL_mm_um", "subL_it_i0", "B: left (column/output) subspace overlap", "overlap"),
              ("subR_mm_um", "subR_it_i0", "B: right (row/input) subspace overlap", "overlap"),
              ("jaccard_mm_um", "jaccard_cross_it", "A: support overlap (Jaccard)", "Jaccard"),
              ("spearman_mm_um", "spearman_cross_it", "A: energy-map correlation (Spearman)", r"$\rho$")]
    n_pair = len(d["per_target"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4))
    cols = _pair_colors(n_pair)
    for ax, (key, ckey, ttl, yl) in zip(axes.ravel(), panels):
        x = np.arange(len(projs))
        w = 0.8 / (n_pair + 1)
        for i, ti in enumerate(range(n_pair)):
            mods = d["per_target"][ti]["modules"]
            y = [np.mean([v[key] for n, v in mods.items() if n.endswith(p)]) for p in projs]
            ax.bar(x + (i - n_pair / 2.0) * w, y, w, color=cols[i], label=f"mm vs um (pair {ti})")
        yc = []
        for p in projs:
            vals = []
            for ti in range(1, n_pair):
                mods = d["per_target"][ti]["modules"]
                vals += [v[ckey] for n, v in mods.items() if n.endswith(p) and ckey in v]
            yc.append(np.mean(vals) if vals else np.nan)
        if not np.all(np.isnan(yc)):
            ax.bar(x + (n_pair - n_pair / 2.0) * w, yc, w, color="#555555",
                   label="IT$_i$ vs IT$_0$ (diff. pair)")
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_proj", "") for p in projs])
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("per-projection breakdown (each layer = mean over its 7 projections in the line plots)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, outdir, "fig_gradsub_proj")
    plt.close(fig)


def fig_B_lp(jsonpath, outdir):
    """B 的 层(32) x 投影(7) 热力图：色 = 子空间重叠（上排 subL/输出侧，下排 subR/输入侧）。"""
    import json
    import re
    d = json.load(open(jsonpath))
    projs = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    n_pair = len(d["per_target"])
    fig, axes = plt.subplots(2, n_pair, figsize=(3.0 * n_pair + 1.6, 8.6), squeeze=False)
    rows = (("subL_mm_um", "subL (left/column-output side)"),
            ("subR_mm_um", "subR (right/row-input side)"))
    ims = {}
    for r, (key, ttl) in enumerate(rows):
        mats, vmax = [], 0.0
        for ti in range(n_pair):
            M = np.full((32, len(projs)), np.nan)
            for n, v in d["per_target"][ti]["modules"].items():
                m = re.search(r"layers\.(\d+)\.", n)
                if not m:
                    continue
                L = int(m.group(1))
                if not 0 <= L < 32:
                    continue
                for j, p in enumerate(projs):
                    if n.endswith(p):
                        M[L, j] = v[key]
            mats.append(M)
            vmax = max(vmax, np.nanmax(M))
        for c, ti in enumerate(range(n_pair)):
            ax = axes[r, c]
            ims[(r, c)] = ax.imshow(mats[c], aspect="auto", origin="lower", cmap="viridis",
                                    vmin=0, vmax=vmax)
            ax.set_xticks(range(len(projs)))
            ax.set_xticklabels([p.replace("_proj", "") for p in projs], fontsize=8)
            ax.set_yticks([0, 8, 16, 24, 31])
            if c == 0:
                ax.set_ylabel("layer")
            ax.set_title(f"{ttl}  (pair {ti})", fontsize=9)
            if r == 1:
                ax.set_xlabel("projection")
    fig.suptitle("B: subspace overlap, layer x projection "
                 "(top = subL/left/output, bottom = subR/right/input)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.90, 0.94))
    cax1 = fig.add_axes([0.915, 0.55, 0.012, 0.33])
    fig.colorbar(ims[(0, n_pair - 1)], cax=cax1).set_label("subL overlap", size=8)
    cax2 = fig.add_axes([0.915, 0.11, 0.012, 0.33])
    fig.colorbar(ims[(1, n_pair - 1)], cax=cax2).set_label("subR overlap", size=8)
    save(fig, outdir, "fig_gradsub_B_lp")
    plt.close(fig)


def fig_cos(jsonpath, outdir):
    """直接梯度余弦：cos(gW_mm, gW_um) 逐层（3 个 pair + 同模态基线 + 0 线）。"""
    import json
    d = json.load(open(jsonpath))
    pl = d["per_layer"]
    layers = sorted(int(k) for k in pl)
    n_pair = len(d["per_target"])
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for ti, col in enumerate(_pair_colors(n_pair)):
        y = [pl[str(L)].get(str(ti), {}).get("cos_raw") for L in layers]
        ax.plot(layers, y, "-o", ms=2.8, lw=1.5, color=col, label=f"mm vs um (pair {ti})")
    yc = _baseline(pl, layers, "cos_raw_cross_it", n_pair)
    if any(v is not None for v in yc):
        ax.plot(layers, yc, "--", color="black", lw=1.3, label="IT$_i$ vs IT$_0$ (diff. pair)")
    ax.axhline(0, ls=":", color="gray", lw=1.2, label="0 = orthogonal")
    ax.set_xlabel("layer index")
    ax.set_ylabel(r"$\cos(gW_{mm}, gW_{um})$")
    ax.set_title("direct per-module gradient cosine (raw gW, answer tokens, base model)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, outdir, "fig_gradsub_cos")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emaps", default="../product/results/_analysis/grad_subspace_emaps.npz")
    ap.add_argument("--json", dest="json_path",
                    default="../product/results/_analysis/grad_subspace.json")
    ap.add_argument("--outdir", default="../references/think/figs")
    ap.add_argument("--layers", default="8,16,24,31")
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    global SUFFIX
    SUFFIX = args.suffix
    z = np.load(args.emaps)
    layers = [int(x) for x in args.layers.split(",")]
    ti = args.target
    cols = [("mm", "self_attn.q_proj"), ("um", "self_attn.q_proj"),
            ("mm", "mlp.down_proj"), ("um", "mlp.down_proj")]

    def find(m, proj, lay):
        key = f"{m}|{ti}|model.language_model.layers.{lay}.{proj}"
        return z[key].astype(np.float32) if key in z else None

    fig, axes = plt.subplots(len(layers), len(cols), figsize=(4 * len(cols), 4 * len(layers)))
    for r, lay in enumerate(layers):
        for c, (m, proj) in enumerate(cols):
            ax = axes[r, c]
            E = find(m, proj, lay)
            if E is None:
                ax.axis("off")
                continue
            tot = E.sum()
            diag = np.diag(E).sum()
            frac = diag / tot if tot > 0 else 0.0
            v = np.log10(E + E.max() * 1e-6 + 1e-30)
            im = ax.imshow(v, cmap="magma", origin="lower", aspect="auto")
            k = E.shape[0]
            ax.plot([0, k - 1], [0, k - 1], "--", color="cyan", lw=0.9, alpha=0.9)
            ax.set_title(f"L{lay} {proj.split('.')[-1]}·{m}\ndiag energy = {frac*100:.4f}%",
                         fontsize=9)
            ax.set_xticks([0, k // 2, k - 1])
            ax.set_yticks([0, k // 2, k - 1])
            ax.tick_params(labelsize=7)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=6)
            cb.set_label(r"$\log_{10}|C|^2$", size=7)
    fig.suptitle(f"$|C|^2$ energy maps in W's SVD basis  (target pair {ti}; cyan dashed = diagonal)\n"
                 "colorbar = log10|C|^2  (negative ticks = SMALL energies;  E = |C|^2 ≥ 0) — "
                 "diag energy ~ K/(d_out·d_in) % means the diagonal is essentially empty",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, args.outdir, "fig_gradsub_emaps")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, proj in zip(axes, ("self_attn.q_proj", "mlp.down_proj")):
        for m, col in (("mm", "#d62728"), ("um", "#1f77b4")):
            for lay, ls in zip(layers, ("-", "--", "-.", ":")):
                E = find(m, proj, lay)
                if E is None:
                    continue
                ax.semilogy(np.arange(E.shape[0]), np.diag(E) + 1e-30,
                            ls, color=col, lw=1.0, alpha=0.8,
                            label=f"{m} L{lay}")
        ax.set_title(f"diagonal profile |diag(C)|$^2$  ({proj.split('.')[-1]})")
        ax.set_xlabel("singular-direction index j of W (top-256)")
        ax.set_ylabel("|diag(C)|$^2$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle(r"diagonal profile $|\mathrm{diag}(C)|^2$ (log axis) — "
                 "carries essentially no energy (near numerical floor)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, args.outdir, "fig_gradsub_diag")
    plt.close(fig)
    fig_A(args.json_path, args.outdir)
    fig_B(args.json_path, args.outdir)
    fig_proj(args.json_path, args.outdir)
    fig_B_lp(args.json_path, args.outdir)
    fig_cos(args.json_path, args.outdir)
    print("saved fig_gradsub_emaps/diag/A_emaps/B_layers to", args.outdir)


if __name__ == "__main__":
    main()
