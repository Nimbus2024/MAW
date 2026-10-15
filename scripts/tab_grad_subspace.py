"""X1 阈值敏感度总表：四个计算方法 x 5 对 x 三档阈值（无阈值 / k=1% / k=5%）。

主口径 = 路线 A：把"同模态异题"当通用/格式对照（**IT 侧与 PT 侧都算**，取两者均值），
报 mm-um 相对该基线的超额倍数。
每格格式：`值 (÷ 基线)`；pair 0 是基线参照对（IT_0 / PT_0），无基线。
"""
import argparse
import json
from pathlib import Path

import numpy as np

METHODS = [
    ("(1) cos_diag  [W-basis diag]", "cos_diag", ("cos_diag_cross_it", "cos_diag_cross_um")),
    ("(2A) Jaccard  [energy-map support]", "jaccard_mm_um", ("jaccard_cross_it", "jaccard_cross_um")),
    ("(2A) Spearman [energy-map rank]", "spearman_mm_um", ("spearman_cross_it", "spearman_cross_um")),
    ("(3B) subL     [left/column]", "subL_mm_um", ("subL_it_i0", "subL_um_i0")),
    ("(3B) subR     [right/row]", "subR_mm_um", ("subR_it_i0", "subR_um_i0")),
    ("(4)  cos_raw  [direct grad cos]", "cos_raw", ("cos_raw_cross_it", "cos_raw_cross_um")),
]
THRESHOLDS = [("100", "no threshold (k=100%)"), ("1", "k = 1%"), ("5", "k = 5%")]
RATIO_MIN_BASE = 0.005   # 基线非正或太小时比值无意义（cos 的异题基线≈0，可负）-> 改用差值


def v_layer(d, ti, key):
    pl = d["per_layer"]
    vs = [pl[L][str(ti)][key] for L in sorted(pl, key=int)
          if key in pl[L].get(str(ti), {}) and pl[L][str(ti)][key] is not None]
    return float(np.mean(vs)) if vs else None


def base_pair(d, ti, bkeys):
    """该对的异题基线 = IT/PT 两侧均值（各自有则纳入）。"""
    vs = [v_layer(d, ti, k) for k in bkeys]
    vs = [v for v in vs if v is not None]
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

    out = ["## 目标对（跳过退化答案：`NA` / 答案 token < 2），seed=42", "",
           "| # | perm_idx | row | 问题 | 答案 | ans tokens |", "|---|---|---|---|---|---|"]
    for ti, t in enumerate(metas):
        m = t.get("meta")
        if m is None:
            out.append(f"| pair {ti} | ? | ? | (no meta) | ? | ? |")
            continue
        q_or_key = m.get("q_um") or m.get("key", "?")
        out.append(f"| pair {ti} | {m['perm_idx']} | {m['row']} | "
                   f"{q_or_key.replace('|', '/')} | {m['ans'].replace('|', '/')[:44]} | "
                   f"{m['ans_tokens']} |")
    out += ["", "基线 = **同模态异题**：`IT_i vs IT_0` 与 `PT_i vs PT_0` 两侧都算、取均值。"
                "比值 = mm-um ÷ 该基线。pair 0 即 IT_0/PT_0，无基线。", ""]

    for tag, tname in THRESHOLDS:
        d = data[tag]
        out += [f"### 阈值：{tname}", "",
                "| 方法 | " + " | ".join(f"pair {i}" for i in range(n_pair)) + " |",
                "|" + "---|" * (n_pair + 1)]
        for label, key, bkeys in METHODS:
            cells = []
            for ti in range(n_pair):
                v = v_layer(d, ti, key)
                if v is None:
                    cells.append("—")
                elif ti == 0:
                    cells.append(f"{v:.4f}")
                else:
                    b = base_pair(d, ti, bkeys)
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
            for _, _, bkeys in METHODS:
                it, um = v_layer(d, ti, bkeys[0]), v_layer(d, ti, bkeys[1])
                parts.append("—" if (it is None and um is None)
                             else f"{(it if it is not None else float('nan')):.4f}/"
                                  f"{(um if um is not None else float('nan')):.4f}")
            bl_cells.append(" ; ".join(parts))
        out += ["| 异题基线 IT/PT (cos_diag / J / rho / subL / subR / cos) | "
                + " | ".join(bl_cells) + " |", ""]

    out += ["### 汇总：÷ 异题基线（行 = 方法 @ 阈值，列 = pair）", "",
            "| 方法 @ 阈值 | " + " | ".join(f"pair {i}" for i in range(1, n_pair)) + " |",
            "|" + "---|" * n_pair]
    for tag, tname in THRESHOLDS:
        d = data[tag]
        for label, key, bkeys in METHODS:
            cells = []
            for ti in range(1, n_pair):
                v, b = v_layer(d, ti, key), base_pair(d, ti, bkeys)
                if v is None or b is None:
                    cells.append("—")
                elif b < RATIO_MIN_BASE:
                    cells.append(f"d{v - b:+.3f}")
                else:
                    cells.append(f"{v / b:.1f}")
            out.append(f"| {label.split('[')[0].strip()} @ {tname} | " + " | ".join(cells) + " |")

    Path(args.out).write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
