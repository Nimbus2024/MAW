#!/usr/bin/env python3
import argparse
import glob
import os
import re


def parse_log(path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    gam = [float(x) for x in re.findall(r"gamma=([0-9.]+)", txt)]
    gap = [float(x) for x in re.findall(r"gap=(-?[0-9.]+)", txt)]
    ge = [float(x) for x in re.findall(r"gap_ema=(-?[0-9.]+)", txt)]
    n = min(len(gam), len(gap), len(ge))
    s = [gap[i] - ge[i] for i in range(n)]
    return gam[:n], gap[:n], s


def polyline(vals, x0, y0, w, h, vmin, vmax):
    n = len(vals)
    if n == 0:
        return ""
    span = (vmax - vmin) or 1.0
    pts = []
    for i, v in enumerate(vals):
        x = x0 + (w * i / max(n - 1, 1))
        y = y0 + h - (h * (v - vmin) / span)
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def panel(title, vals, x0, y0, w, h, color):
    vmin, vmax = min(vals), max(vals)
    pad = (vmax - vmin) * 0.1 or 0.5
    vmin -= pad
    vmax += pad
    out = [f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#999"/>']
    if vmin < 0 < vmax:
        yzero = y0 + h - (h * (0 - vmin) / (vmax - vmin))
        out.append(f'<line x1="{x0}" y1="{yzero:.1f}" x2="{x0+w}" y2="{yzero:.1f}" stroke="#ccc" stroke-dasharray="3,3"/>')
    out.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.4" points="{polyline(vals, x0, y0, w, h, vmin, vmax)}"/>')
    out.append(f'<text x="{x0}" y="{y0-6}" font-size="12">{title}  min={min(vals):.3f} max={max(vals):.3f} mean={sum(vals)/len(vals):.3f}</text>')
    return "\n".join(out)


def build_svg(tag, gam, gap, s):
    W, H = 900, 780
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
             f'<text x="20" y="24" font-size="16">{tag}</text>']
    parts.append(panel("gamma", gam, 40, 50, 820, 180, "#1f77b4"))
    parts.append(panel("gap", gap, 40, 300, 820, 180, "#d62728"))
    parts.append(panel("s = gap - gap_ema", s, 40, 550, 820, 180, "#2ca02c"))
    parts.append("</svg>")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../product")
    ap.add_argument("--sub", default="ablation")
    ap.add_argument("--out", default="ablations/curves")
    ap.add_argument("--tags", default="")
    args = ap.parse_args()
    root = os.path.join(args.root, "results", args.sub)
    want = [t for t in args.tags.split(",") if t.strip()]
    outdir = os.path.join(args.root, "record", args.out)
    os.makedirs(outdir, exist_ok=True)
    logs = sorted(glob.glob(os.path.join(root, "*", "*-*", "logs", "stdout.log")))
    made = 0
    for log in logs:
        tag = os.path.basename(os.path.dirname(os.path.dirname(log)))
        tag = tag.split("-", 1)[1] if "-" in tag else tag
        if want and tag not in want:
            continue
        gam, gap, s = parse_log(log)
        if not gam:
            continue
        with open(os.path.join(outdir, f"{tag}.svg"), "w", encoding="utf-8") as f:
            f.write(build_svg(tag, gam, gap, s))
        made += 1
        print(f"written {outdir}/{tag}.svg  (n={len(gam)})")
    print(f"total {made}")


if __name__ == "__main__":
    main()
