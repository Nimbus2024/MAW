#!/usr/bin/env python3
import argparse
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_ablation_groups as P

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F_TITLE = ImageFont.truetype(BOLD_PATH, 30)
F_PANEL = ImageFont.truetype(BOLD_PATH, 22)
F_AXIS = ImageFont.truetype(FONT_PATH, 17)
F_TICK = ImageFont.truetype(FONT_PATH, 16)
F_LEG = ImageFont.truetype(FONT_PATH, 19)

EN_TITLES = {
    "G1_dynamic_vs_static": "G1  dynamic gamma vs static gamma=0.5",
    "G2_ema_rho": "G2  EMA smoothing rho (0.5 / 0.8 / 0.9 / 1 = no EMA)",
    "G3_beta": "G3  beta (0.1 -> 0.8)",
    "G4_alpha": "G4  alpha (0.5 -> 3)",
    "G5_coeff": "G5  coefficient: inv_beta vs one (DPO)",
    "A0_budget": "A0  budget calibration (lr x steps)",
}
PANELS = [
    ("Forget Fill (All)  lower = more forgotten",
     lambda d: P.series_metric(d, "Forget", "Fill")),
    ("Retain Fill (All)  higher = better kept",
     lambda d: P.series_metric(d, "Retain", "Fill")),
    ("Forget Fill IT - PT  (positive = IT less forgotten)",
     lambda d: (P.series_metric(d, "Forget", "Fill", "IT")
                - P.series_metric(d, "Forget", "Fill", "PT"))
               if P.series_metric(d, "Forget", "Fill", "IT") is not None
               and P.series_metric(d, "Forget", "Fill", "PT") is not None else None),
    ("Retain Fill IT - PT  (positive = IT better kept)",
     lambda d: (P.series_metric(d, "Retain", "Fill", "IT")
                - P.series_metric(d, "Retain", "Fill", "PT"))
               if P.series_metric(d, "Retain", "Fill", "IT") is not None
               and P.series_metric(d, "Retain", "Fill", "PT") is not None else None),
]


def draw_panel(draw, x0, y0, w, h, title, series, color_of):
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(255, 255, 255), outline=(120, 120, 120), width=2)
    draw.text((x0, y0 - 30), title, fill=(0, 0, 0), font=F_PANEL)
    vals = [v for _, ys in series for v in ys if v is not None]
    if not vals:
        return
    vmin, vmax = min(vals), max(vals)
    pad = (vmax - vmin) * 0.15 or 1.0
    vmin -= pad
    vmax += pad
    n = max((len(ys) for _, ys in series), default=1)
    for k in range(5):
        yv = vmin + (vmax - vmin) * k / 4
        yy = y0 + h - h * k / 4
        draw.line([x0, yy, x0 + w, yy], fill=(235, 235, 235))
        draw.text((x0 - 78, yy - 10), f"{yv:.2f}", fill=(90, 90, 90), font=F_TICK)
    for i in range(n):
        xx = x0 + w * i / max(n - 1, 1)
        draw.line([xx, y0, xx, y0 + h], fill=(245, 245, 245))
        draw.text((xx - 5, y0 + h + 8), str(i + 1), fill=(90, 90, 90), font=F_TICK)
    draw.text((x0 + w / 2 - 22, y0 + h + 30), "epoch", fill=(60, 60, 60), font=F_AXIS)
    for tag, ys in series:
        col = color_of(tag)
        pts = []
        for i, v in enumerate(ys):
            if v is None:
                continue
            xx = x0 + w * i / max(n - 1, 1)
            yy = y0 + h - h * (v - vmin) / (vmax - vmin)
            pts.append((xx, yy))
        if len(pts) >= 2:
            draw.line(pts, fill=col, width=4)
        for pt in pts:
            draw.ellipse([pt[0] - 5, pt[1] - 5, pt[0] + 5, pt[1] + 5], fill=col, outline=(255, 255, 255))
        if pts:
            lx, ly = pts[-1]
            draw.text((lx + 8, ly - 10), f"{ys[-1]:.1f}", fill=col, font=F_TICK)


def render_family(name, title, tags, runs, outdir):
    color_of = lambda tag: P.COLORS[tags.index(tag) % len(P.COLORS)]
    present = [t for t in tags if t in runs]
    img = Image.new("RGB", (1960, 1560), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 22), EN_TITLES.get(name, title), fill=(0, 0, 0), font=F_TITLE)
    lx, ly = 30, 78
    for tag in present:
        draw.line([lx, ly + 9, lx + 34, ly + 9], fill=color_of(tag), width=5)
        draw.text((lx + 42, ly), tag, fill=(0, 0, 0), font=F_LEG)
        lx += 60 + int(draw.textlength(tag, font=F_LEG))
        if lx > 1500:
            lx, ly = 30, ly + 34
    y_base = ly + 50
    positions = [(110, y_base), (1010, y_base), (110, y_base + 660), (1010, y_base + 660)]
    for (label, fn), (x0, y0) in zip(PANELS, positions):
        series = []
        for tag in present:
            eps = runs[tag]["epochs"]
            ys = [fn(eps[e]) if e in eps else None for e in sorted(eps)]
            if any(v is not None for v in ys):
                series.append((tag, ys))
        draw_panel(draw, x0, y0, 820, 500, label, series, color_of)
    path = os.path.join(outdir, f"{name}.png")
    img.save(path)
    return path


def render_controller(tag, logpath, outdir):
    txt = open(logpath, encoding="utf-8", errors="ignore").read()
    gam = [float(x) for x in re.findall(r"gamma=([0-9.]+)", txt)]
    gap = [float(x) for x in re.findall(r"gap=(-?[0-9.]+)", txt)]
    ge = [float(x) for x in re.findall(r"gap_ema=(-?[0-9.]+)", txt)]
    n = min(len(gam), len(gap), len(ge))
    s = [gap[i] - ge[i] for i in range(n)]
    img = Image.new("RGB", (1960, 1100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 22), f"controller signals - {tag}", fill=(0, 0, 0), font=F_TITLE)
    specs = [("gamma", gam[:n], (31, 119, 180)), ("gap", gap[:n], (214, 39, 40)),
             ("s = gap - gap_ema", s, (44, 160, 44))]
    for i, (label, ys, col) in enumerate(specs):
        draw_panel(draw, 110, 100 + i * 330, 1760, 250, label, [(tag, ys)], lambda _t, c=col: c)
    path = os.path.join(outdir, f"{tag}_controller.png")
    img.save(path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../product")
    args = ap.parse_args()
    root = os.path.join(args.root, "results", "ablation")
    runs = P.load_runs(root)
    gdir = os.path.join(args.root, "record", "ablations", "groups")
    cdir = os.path.join(args.root, "record", "ablations", "curves")
    os.makedirs(gdir, exist_ok=True)
    os.makedirs(cdir, exist_ok=True)
    for name, (title, tags) in P.FAMILIES.items():
        print("written", render_family(name, title, tags, runs, gdir))
    for tag in ("v1ema", "v6a3"):
        for log in P.glob.glob(os.path.join(root, "*", "*", "logs", "stdout.log")):
            ts = os.path.basename(os.path.dirname(os.path.dirname(log)))
            t = ts.split("-", 1)[1] if "-" in ts else ts
            if t == tag:
                print("written", render_controller(tag, log, cdir))
                break


if __name__ == "__main__":
    main()
