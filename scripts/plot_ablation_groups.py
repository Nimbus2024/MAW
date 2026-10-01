#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_record as G

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
          "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22"]

FAMILIES = {
    "G1_dynamic_vs_static": ("动态 γ vs 静态 γ=0.5（DPO/beta 设定）",
                             ["v1ema", "v3fix05"]),
    "G2_ema_rho": ("动态 γ 的 EMA 平滑程度 ρ（0.5 / 0.8 / 0.9 / 1=无 EMA）",
                   ["v7r05", "v1ema", "v7r09", "v2gap"]),
    "G3_beta": ("β 从小到大",
                ["v1ema", "v8b02", "v8b04", "v8b08"]),
    "G4_alpha": ("α 从小到大",
                 ["v6a05", "v1ema", "v6a15", "v6a3"]),
    "G5_coeff": ("DPO 是否除以 β（系数 inv_beta vs one）",
                 ["v3fix05", "v9coeff1", "v1ema", "v10dpo"]),
    "A0_budget": ("附录：预算校准（lr × steps）",
                  ["cal_lr5em6_s50", "cal_lr5em6_s95", "cal_lr75em6_s75",
                   "cal_lr75em6_s95", "cal_lr1em5_s50", "cal_lr1em5_s95",
                   "cal_lr25em6_s150", "cal_lr25em6_s200"]),
}


def load_runs(root):
    runs = {}
    for log in sorted(glob.glob(os.path.join(root, "*", "*", "logs", "stdout.log"))):
        run = os.path.dirname(os.path.dirname(log))
        ts = os.path.basename(run)
        tag = ts.split("-", 1)[1] if "-" in ts else ts
        cfg = {}
        cf = os.path.join(run, "config", "args.json")
        if os.path.isfile(cf):
            cfg = json.load(open(cf))
        epochs = {}
        for ep_dir in glob.glob(os.path.join(run, "runs", "epoch-*")):
            ep = int(re.search(r"epoch-(\d+)", ep_dir).group(1))
            js = glob.glob(os.path.join(ep_dir, "metrics", "*final_evaluation_results.json"))
            if js:
                epochs[ep] = G.parse_final(os.path.dirname(js[0]))
        txt = open(log, encoding="utf-8", errors="ignore").read()
        gam = [float(x) for x in re.findall(r"gamma=([0-9.]+)", txt)]
        runs[tag] = {"cfg": cfg, "epochs": epochs, "gamma": gam}
    return runs


def series_metric(data, group, task, modal="All"):
    it, pt, al = G._task_vals(data, group, task)
    return {"IT": it, "PT": pt, "All": al}[modal]


def panel(series, xlabel, ylabel, x0, y0, w, h):
    allv = [v for _, ys in series for v in ys if v is not None]
    if not allv:
        return ""
    vmin, vmax = min(allv), max(allv)
    pad = (vmax - vmin) * 0.12 or 1.0
    vmin -= pad
    vmax += pad
    n = max((len(ys) for _, ys in series), default=1)
    out = [f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fcfcfc" stroke="#999"/>',
           f'<text x="{x0}" y="{y0-8}" font-size="13">{ylabel}</text>']
    if vmin < 0 < vmax:
        yz = y0 + h - h * (0 - vmin) / (vmax - vmin)
        out.append(f'<line x1="{x0}" y1="{yz:.1f}" x2="{x0+w}" y2="{yz:.1f}" stroke="#ddd" stroke-dasharray="3,3"/>')
    for i in range(1, 4):
        x = x0 + w * i / 4
        out.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+h}" stroke="#eee"/>')
    for idx, (tag, ys) in enumerate(series):
        pts = []
        for i, v in enumerate(ys):
            if v is None:
                continue
            x = x0 + w * i / max(n - 1, 1)
            y = y0 + h - h * (v - vmin) / (vmax - vmin)
            pts.append(f"{x:.1f},{y:.1f}")
        if pts:
            out.append(f'<polyline fill="none" stroke="{COLORS[idx % len(COLORS)]}" stroke-width="1.6" points="{" ".join(pts)}"/>')
    for i in range(n):
        x = x0 + w * i / max(n - 1, 1)
        out.append(f'<text x="{x-4:.1f}" y="{y0+h+14}" font-size="10">{i+1}</text>')
    out.append(f'<text x="{x0+w-60}" y="{y0+h+14}" font-size="10">{xlabel}</text>')
    return "\n".join(out)


def legend(entries, x, y):
    out = []
    for i, tag in enumerate(entries):
        yy = y + i * 16
        out.append(f'<rect x="{x}" y="{yy-9}" width="12" height="8" fill="{COLORS[i % len(COLORS)]}"/>')
        out.append(f'<text x="{x+16}" y="{yy}" font-size="11">{G.esc(tag)}</text>')
    return "\n".join(out)


def build_family_svg(name, title, tags, runs, outdir):
    series_specs = [
        ("Forget Fill (All)", lambda d: series_metric(d, "Forget", "Fill")),
        ("Retain Fill (All)", lambda d: series_metric(d, "Retain", "Fill")),
        ("Forget Fill IT - PT", lambda d: (
            series_metric(d, "Forget", "Fill", "IT") - series_metric(d, "Forget", "Fill", "PT")
            if series_metric(d, "Forget", "Fill", "IT") is not None
            and series_metric(d, "Forget", "Fill", "PT") is not None else None)),
        ("Retain Fill IT - PT", lambda d: (
            series_metric(d, "Retain", "Fill", "IT") - series_metric(d, "Retain", "Fill", "PT")
            if series_metric(d, "Retain", "Fill", "IT") is not None
            and series_metric(d, "Retain", "Fill", "PT") is not None else None)),
    ]
    W, H = 1000, 900
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
             f'<text x="20" y="24" font-size="16">{G.esc(title)}</text>']
    parts.append(legend([t for t in tags if t in runs], 700, 40))
    panels = []
    positions = [(50, 60), (520, 60), (50, 330), (520, 330)]
    for (label, fn), (x0, y0) in zip(series_specs, positions):
        series = []
        for tag in tags:
            if tag not in runs:
                continue
            eps = runs[tag]["epochs"]
            ys = [fn(eps[e]) if e in eps else None for e in sorted(eps)]
            if any(v is not None for v in ys):
                series.append((tag, ys))
        panels.append(panel(series, "epoch", label, x0, y0, 420, 220))
    parts += panels
    parts.append("</svg>")
    path = os.path.join(outdir, f"{name}.svg")
    return "\n".join(parts), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../product")
    ap.add_argument("--out", default="ablations/groups")
    args = ap.parse_args()
    root = os.path.join(args.root, "results", "ablation")
    outdir = os.path.join(args.root, "record", args.out)
    os.makedirs(outdir, exist_ok=True)
    runs = load_runs(root)
    print("loaded", len(runs), "runs")
    for name, (title, tags) in FAMILIES.items():
        svg, path = build_family_svg(name, title, tags, runs, outdir)
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        print("written", path)
        for tag in tags:
            if tag not in runs:
                print("  missing", tag)
                continue
            eps = sorted(runs[tag]["epochs"])
            last = eps[-1]
            d = runs[tag]["epochs"][last]
            fc = series_metric(d, "Forget", "Classif")
            rf = series_metric(d, "Retain", "Fill")
            rl = series_metric(d, "Real", "Fill")
            print(f"  {tag:16s} ep{last} fCls={fc:.1f} rFill={rf:.1f} realFill={rl:.1f}")


if __name__ == "__main__":
    main()
