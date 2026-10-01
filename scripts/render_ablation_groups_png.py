#!/usr/bin/env python3
import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_ablation_groups as P

FONT = ImageFont.load_default()


def draw_panel(draw, x0, y0, w, h, title, series, color_of):
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(252, 252, 252), outline=(150, 150, 150))
    vals = [v for _, ys in series for v in ys if v is not None]
    if not vals:
        return
    vmin, vmax = min(vals), max(vals)
    pad = (vmax - vmin) * 0.12 or 1.0
    vmin -= pad
    vmax += pad
    n = max((len(ys) for _, ys in series), default=1)
    draw.text((x0 + 4, y0 - 14), title, fill=(0, 0, 0), font=FONT)
    draw.text((x0 + 4, y0 + 2), f"max={max(vals):.2f}", fill=(120, 120, 120), font=FONT)
    draw.text((x0 + 4, y0 + h - 12), f"min={min(vals):.2f}", fill=(120, 120, 120), font=FONT)
    if vmin < 0 < vmax:
        yz = y0 + h - h * (0 - vmin) / (vmax - vmin)
        draw.line([x0, yz, x0 + w, yz], fill=(220, 220, 220))
    for i in range(1, 4):
        xx = x0 + w * i / 4
        draw.line([xx, y0, xx, y0 + h], fill=(238, 238, 238))
    for tag, ys in series:
        pts = []
        for i, v in enumerate(ys):
            if v is None:
                continue
            xx = x0 + w * i / max(n - 1, 1)
            yy = y0 + h - h * (v - vmin) / (vmax - vmin)
            pts.append((xx, yy))
        if len(pts) >= 2:
            draw.line(pts, fill=color_of(tag), width=2)
        for pt in pts:
            draw.ellipse([pt[0] - 2, pt[1] - 2, pt[0] + 2, pt[1] + 2], fill=color_of(tag))
    for i in range(n):
        xx = x0 + w * i / max(n - 1, 1)
        draw.text((xx - 3, y0 + h + 4), str(i + 1), fill=(80, 80, 80), font=FONT)
    draw.text((x0 + w - 44, y0 + h + 4), "epoch", fill=(80, 80, 80), font=FONT)


def render_family(name, title, tags, runs, outdir):
    color_of = lambda tag: P.COLORS[tags.index(tag) % len(P.COLORS)]
    specs = [
        ("Forget Classif (All, lower=forgotten)", lambda d: P.series_metric(d, "Forget", "Classif")),
        ("Retain Fill (All, higher=kept)", lambda d: P.series_metric(d, "Retain", "Fill")),
        ("Real Fill (All, higher=better)", lambda d: P.series_metric(d, "Real", "Fill")),
        ("|Forget Cls IT - PT| (modal imbalance)", lambda d: (
            abs(P.series_metric(d, "Forget", "Classif", "IT") - P.series_metric(d, "Forget", "Classif", "PT"))
            if P.series_metric(d, "Forget", "Classif", "IT") is not None
            and P.series_metric(d, "Forget", "Classif", "PT") is not None else None)),
    ]
    img = Image.new("RGB", (1320, 1000), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((24, 14), title, fill=(0, 0, 0), font=FONT)
    for i, tag in enumerate([t for t in tags if t in runs]):
        yy = 34 + i * 13
        draw.line([1000, yy + 4, 1020, yy + 4], fill=color_of(tag), width=3)
        draw.text((1026, yy), tag, fill=(0, 0, 0), font=FONT)
    positions = [(60, 90), (700, 90), (60, 560), (700, 560)]
    for (label, fn), (x0, y0) in zip(specs, positions):
        series = []
        for tag in tags:
            if tag not in runs:
                continue
            eps = runs[tag]["epochs"]
            ys = [fn(eps[e]) if e in eps else None for e in sorted(eps)]
            if any(v is not None for v in ys):
                series.append((tag, ys))
        draw_panel(draw, x0, y0, 560, 400, label, series, color_of)
    path = os.path.join(outdir, f"{name}.png")
    img.save(path)
    return path


def render_controller(tag, runs, outdir):
    r = runs[tag]
    import re
    txt = None
    for log in P.glob.glob(os.path.join(args_root[0], "*", "*", "logs", "stdout.log")):
        ts = os.path.basename(os.path.dirname(os.path.dirname(log)))
        t = ts.split("-", 1)[1] if "-" in ts else ts
        if t == tag:
            txt = open(log, encoding="utf-8", errors="ignore").read()
            break
    if not txt:
        return None
    gam = [float(x) for x in re.findall(r"gamma=([0-9.]+)", txt)]
    gap = [float(x) for x in re.findall(r"gap=(-?[0-9.]+)", txt)]
    ge = [float(x) for x in re.findall(r"gap_ema=(-?[0-9.]+)", txt)]
    n = min(len(gam), len(gap), len(ge))
    s = [gap[i] - ge[i] for i in range(n)]
    img = Image.new("RGB", (1320, 700), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((24, 14), f"{tag}: controller signals", fill=(0, 0, 0), font=FONT)
    for i, (label, ys, col) in enumerate((("gamma", gam, (31, 119, 180)), ("gap", gap, (214, 39, 40)), ("s=gap-gap_ema", s, (44, 160, 44)))):
        draw_panel(draw, 60, 80 + i * 210, 1180, 170, label, [(tag, ys[:n])], lambda _t, c=col: c)
    path = os.path.join(outdir, f"{tag}_controller.png")
    img.save(path)
    return path


args_root = ["../product"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../product")
    args = ap.parse_args()
    args_root[0] = os.path.join(args.root, "results", "ablation")
    runs = P.load_runs(args_root[0])
    gdir = os.path.join(args.root, "record", "ablations", "groups")
    os.makedirs(gdir, exist_ok=True)
    for name, (title, tags) in P.FAMILIES.items():
        path = render_family(name, title, tags, runs, gdir)
        print("written", path)
    cdir = os.path.join(args.root, "record", "ablations", "curves")
    os.makedirs(cdir, exist_ok=True)
    for tag in ("v1ema", "v6a3", "v7r09"):
        p = render_controller(tag, runs, cdir)
        if p:
            print("written", p)
