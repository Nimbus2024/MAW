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


def v_layer(d, ti, key):
    pl = d["per_layer"]
    vs = [pl[L][str(ti)][key] for L in sorted(pl, key=int)
          if key in pl[L].get(str(ti), {}) and pl[L][str(ti)][key] is not None]
    return float(np.mean(vs)) if vs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../product/results/_analysis")
    ap.add_argument("--prefix", default="grad_subspace_k")
    ap.add_argument("--out", default="../references/think/grad_subspace_thresholds.md")
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
                    cells.append(f"{v:.4f} ({v / b:.1f}x)" if b else f"{v:.4f}")
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
                cells.append(f"{v / b:.1f}" if v is not None and b else "—")
            out.append(f"| {label.split('[')[0].strip()} @ {tname} | " + " | ".join(cells) + " |")

    Path(args.out).write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
