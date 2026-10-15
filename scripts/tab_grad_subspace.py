"""X1 阈值敏感度总表：四个计算方法 x 5 对 x 三档阈值（无阈值 / k=1% / k=5%）。

主口径 = 路线 A：把"同模态异题"当通用/格式对照，报 mm-um 相对该基线的超额倍数。
每格格式：`值 (÷ 自己那一对的异题基线)`；pair 0 是基线参照对（IT_0），无基线。
"""
import argparse
import json
from pathlib import Path

import numpy as np

METHODS = [
    ("(1) cos_diag  [W-basis diag]", "cos_diag", None),
    ("(2A) Jaccard  [energy-map support]", "jaccard_mm_um", "jaccard_cross_it"),
    ("(2A) Spearman [energy-map rank]", "spearman_mm_um", "spearman_cross_it"),
    ("(3B) subL     [left/column]", "subL_mm_um", "subL_it_i0"),
    ("(3B) subR     [right/row]", "subR_mm_um", "subR_it_i0"),
    ("(4)  cos_raw  [direct grad cos]", "cos_raw", "cos_raw_cross_it"),
]
THRESHOLDS = [("100", "no threshold (k=100%)"), ("1", "k = 1%"), ("5", "k = 5%")]
RATIO_MIN_BASE = 0.005   # 基线非正或太小时比值无意义（cos 的异题基线≈0，可负）-> 改用差值


def v_layer(d, ti, key):
    pl = d["per_layer"]
    vs = [pl[L][str(ti)][key] for L in sorted(pl, key=int)
          if key in pl[L].get(str(ti), {}) and pl[L][str(ti)][key] is not None]
    return float(np.mean(vs)) if vs else None


def make_fig(data, n_pair, figdir):
    """同一方法在 3 档阈值下的"相对异题基线倍数"，看结论是否随阈值漂移。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mets = [(l.split("[")[0].strip(), k, bk) for l, k, bk in METHODS if bk]
    cols = list(plt.get_cmap("tab10").colors)
    fig, axes = plt.subplots(1, len(mets), figsize=(3.1 * len(mets), 3.6), squeeze=False)
    for ax, (label, key, bk) in zip(axes[0], mets):
        for j, (tag, tname) in enumerate(THRESHOLDS):
            d = data[tag]
            ys = []
            for ti in range(1, n_pair):
                v, b = v_layer(d, ti, key), v_layer(d, ti, bk)
                ys.append(np.nan if (v is None or b is None or b < RATIO_MIN_BASE) else v / b)
            ax.plot(range(1, n_pair), ys, "-o", ms=4, color=cols[j], label=tname)
        ax.axhline(1.0, ls=":", color="gray", lw=1.2)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("target pair")
        ax.set_xticks(range(1, n_pair))
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("mm-um / (same-modality, diff-question)")
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("route-A excess vs threshold  (1.0 = same as the generic/format control)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("svg", "png"):
        fig.savefig(Path(figdir) / f"fig_gradsub_thr_sensitivity.{ext}", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../product/results/_analysis")
    ap.add_argument("--prefix", default="grad_subspace_k")
    ap.add_argument("--out", default="../references/think/grad_subspace_thresholds.md")
    ap.add_argument("--figdir", default="")   # 留空则不出图（图统一用 plot_grad_subspace.py）
    args = ap.parse_args()

    data, metas = {}, None
    for tag, _ in THRESHOLDS:
        f = Path(args.dir) / f"{args.prefix}{tag}.json"
        if not f.exists():
            print(f"!! missing {f}")
            return
        data[tag] = json.load(open(f))
        if metas is None:
            metas = data[tag]["per_target"]
    n_pair = len(metas)

    out = ["## 目标对（跳过退化答案：`NA` / 答案 token < 2），seed=42", ""]
    out += ["| # | perm_idx | row | key | 答案 | ans tokens |", "|---|---|---|---|---|---|"]
    for ti, t in enumerate(metas):
        m = t.get("meta")
        if m is None:
            out.append(f"| pair {ti} | ? | ? | ? | (no meta) | ? |")
            continue
        out.append(f"| pair {ti} | {m['perm_idx']} | {m['row']} | {m['key']} | "
                   f"{m['ans'].replace('|', '/')[:44]} | {m['ans_tokens']} |")
    out += ["", "基线 = **同模态异题**（该对 IT_i vs IT_0）= 通用/格式对照；"
                "比值 = mm-um ÷ 该基线。pair 0 即 IT_0，无基线。", ""]

    for tag, tname in THRESHOLDS:
        d = data[tag]
        out += [f"### 阈值：{tname}", "",
                "| 方法 | " + " | ".join(f"pair {i}" for i in range(n_pair)) + " |",
                "|" + "---|" * (n_pair + 1)]
        for label, key, bkey in METHODS:
            cells = []
            for ti in range(n_pair):
                v = v_layer(d, ti, key)
                if v is None:
                    cells.append("—")
                elif ti == 0 or bkey is None:
                    cells.append(f"{v:.4f}")
                else:
                    b = v_layer(d, ti, bkey)
                    if b is None:
                        cells.append(f"{v:.4f}")
                    elif b < RATIO_MIN_BASE:
                        cells.append(f"{v:.4f} (d={v - b:+.4f})")
                    else:
                        cells.append(f"{v:.4f} ({v / b:.1f}x)")
            out.append("| " + label + " | " + " | ".join(cells) + " |")
        bl_cells = []
        for ti in range(n_pair):
            if ti == 0:
                bl_cells.append("—")
                continue
            parts = []
            for _, _, bkey in METHODS:
                if not bkey:
                    continue
                b = v_layer(d, ti, bkey)
                parts.append(f"{b:.4f}" if b is not None else "—")
            bl_cells.append(" / ".join(parts))
        out += ["| 异题基线 (J / rho / subL / subR / cos) | " + " | ".join(bl_cells) + " |", ""]

    out += ["### 汇总：÷ 异题基线（行 = 方法 @ 阈值，列 = pair）", "",
            "| 方法 @ 阈值 | " + " | ".join(f"pair {i}" for i in range(1, n_pair)) + " |",
            "|" + "---|" * n_pair]
    for tag, tname in THRESHOLDS:
        d = data[tag]
        for label, key, bkey in METHODS:
            if bkey is None:
                continue
            cells = []
            for ti in range(1, n_pair):
                v, b = v_layer(d, ti, key), v_layer(d, ti, bkey)
                if v is None or b is None:
                    cells.append("—")
                elif b < RATIO_MIN_BASE:
                    cells.append(f"d{v - b:+.3f}")
                else:
                    cells.append(f"{v / b:.1f}")
            out.append(f"| {label.split('[')[0].strip()} @ {tname} | " + " | ".join(cells) + " |")

    if args.figdir:
        make_fig(data, n_pair, args.figdir)
    Path(args.out).write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
