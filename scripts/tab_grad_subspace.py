"""X1 阈值敏感度总表：四个计算方法 x 5 对 x 三档阈值（无阈值 / k=1% / k=5%）。

主口径 = 路线 A：把"同模态异题"当通用/格式对照，报 mm-um 相对该基线的超额倍数。
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
RAND = {"subL_mm_um": "subL_rand", "subR_mm_um": "subR_rand"}
THRESHOLDS = [("100", "no threshold (k=100%)"), ("1", "k = 1%"), ("5", "k = 5%")]


def mean_over_layers(d, ti, key):
    pl = d["per_layer"]
    ls = sorted(pl, key=int)
    v = [pl[L][str(ti)][key] for L in ls
         if key in pl[L].get(str(ti), {}) and pl[L][str(ti)][key] is not None]
    return float(np.mean(v)) if v else None


def baseline(d, key, n_pair):
    vs = [mean_over_layers(d, ti, key) for ti in range(1, n_pair)]
    vs = [v for v in vs if v is not None]
    return float(np.mean(vs)) if vs else None


def fmt(v, nd=4):
    return "—" if v is None else f"{v:+.{nd}f}" if v < 0 or v == 0 else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../product/results/_analysis")
    ap.add_argument("--prefix", default="grad_subspace_k")
    ap.add_argument("--out", default="../references/think/grad_subspace_thresholds.md")
    ap.add_argument("--figdir", default="../references/think/figs")
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

    # ---------- 目标对身份 ----------
    lines = ["## 目标对（已剔除退化答案；`NA`/答案 token < 2 跳过）", ""]
    lines += ["| # | perm_idx | row | key | 答案 | ans tokens |",
              "|---|---|---|---|---|---|"]
    for ti, t in enumerate(metas):
        m = t["meta"]
        a = m["ans"].replace("|", "\\|")[:44]
        lines.append(f"| pair {ti} | {m['perm_idx']} | {m['row']} | {m['key']} | {a} | {m['ans_tokens']} |")
    lines += ["", "基线 = 同模态异题（pair 1..4 均值）：**通用/格式**对照。比值 = mm-um ÷ 基线。", ""]

    # ---------- 每档阈值一张表 ----------
    tables = {}
    for tag, tname in THRESHOLDS:
        d = data[tag]
        bl = {bk: baseline(d, k, n_pair) for _, k, bk in METHODS if bk}
        head = "| 方法 | " + " | ".join(f"pair {i}" for i in range(n_pair)) + " | 基线(异题) |"
        sep = "|" + "---|" * (n_pair + 2)
        rows = [head, sep]
        ratio_rows = [head, sep]
        for label, key, bkey in METHODS:
            vals = [mean_over_layers(d, ti, key) for ti in range(n_pair)]
            b = bl.get(bkey)
            rows.append("| " + label + " | " + " | ".join(fmt(v) for v in vals) +
                        " | " + (fmt(b) if b is not None else "—") + " |")
            ratios = []
            for ti in range(n_pair):
                v = vals[ti]
                if ti == 0 or v is None or not b:
                    ratios.append("—")
                else:
                    ratios.append(f"**{v / b:.1f}x**" if v / b >= 1 else f"{v / b:.1f}x")
            ratio_rows.append("| " + label + " | " + " | ".join(ratios) + " | — |")
        tables[tag] = (rows, ratio_rows)

    out = list(lines)
    for tag, tname in THRESHOLDS:
        rows, ratio_rows = tables[tag]
        out += [f"### 阈值：{tname}", "", "**数值**", ""] + rows + ["", "**÷ 异题基线（路线 A 主口径）**", ""] + ratio_rows + [""]

    # ---------- 汇总：比值 x 方法 x 阈值 ----------
    out += ["### 汇总：相对异题基线的倍数（行 = 方法@阈值，列 = pair）", "",
            "| 方法 @ 阈值 | " + " | ".join(f"pair {i}" for i in range(1, n_pair)) + " |",
            "|" + "---|" * n_pair]
    for tag, tname in THRESHOLDS:
        d = data[tag]
        bl = {bk: baseline(d, k, n_pair) for _, k, bk in METHODS if bk}
        for label, key, bkey in METHODS:
            if not bkey:
                continue
            b = bl.get(bkey)
            cells = []
            for ti in range(1, n_pair):
                v = mean_over_layers(d, ti, key)
                cells.append("—" if v is None or not b else f"{v / b:.1f}")
            out.append(f"| {label.split('[')[0].strip()} @ {tname} | " + " | ".join(cells) + " |")

    Path(args.out).write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
