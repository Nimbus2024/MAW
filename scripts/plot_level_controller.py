#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F_TITLE = ImageFont.truetype(BOLD, 30)
F_PANEL = ImageFont.truetype(BOLD, 22)
F_AXIS = ImageFont.truetype(FONT, 17)
F_TICK = ImageFont.truetype(FONT, 16)
F_LEG = ImageFont.truetype(FONT, 19)

COLORS = ["#7f7f7f", "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
SERIES = ["v3fix05", "v1ema", "emaB_r05_a0.5", "emaB_r05_a1.0", "emaB_r05_a2.0", "emaB_r00_a2.0"]


def find_run(root, tag):
    hits = [d for d in glob.glob(os.path.join(root, "*", "*")) if os.path.basename(d).endswith("-" + tag)]
    return hits[0] if hits else None


def load(run):
    cfg = {}
    cf = os.path.join(run, "config", "args.json")
    if os.path.isfile(cf):
        cfg = json.load(open(cf))
    txt = open(os.path.join(run, "logs", "stdout.log"), encoding="utf-8", errors="ignore").read()
    gam = [float(x) for x in re.findall(r"gamma=([0-9.]+)", txt)]
    gap = [float(x) for x in re.findall(r"gap=(-?[0-9.]+)", txt)]
    steps = int(cfg.get("max_steps") or 0) or None
    return cfg, gam, gap, steps


def resample(seq, n):
    if not seq or not n or len(seq) == n:
        return seq
    return [seq[round(i * (len(seq) - 1) / (n - 1))] for i in range(n)]


def level_shat(gap, rho):
    s = 0.0
    out = []
    for t, g in enumerate(gap, start=1):
        s = rho * s + (1.0 - rho) * g
        denom = 1.0 - rho ** t
        out.append(s / denom if denom > 1e-8 else g)
    return out


def panel(draw, x0, y0, w, h, title, series, color_of, vmin=None, vmax=None):
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(255, 255, 255), outline=(120, 120, 120), width=2)
    draw.text((x0, y0 - 30), title, fill=(0, 0, 0), font=F_PANEL)
    vals = [v for _, ys in series for v in ys if v is not None]
    if not vals:
        return
    if vmin is None:
        vmin, vmax = min(vals), max(vals)
        pad = (vmax - vmin) * 0.12 or 0.05
        vmin -= pad
        vmax += pad
    n = max((len(ys) for _, ys in series), default=1)
    for k in range(5):
        yv = vmin + (vmax - vmin) * k / 4
        yy = y0 + h - h * k / 4
        draw.line([x0, yy, x0 + w, yy], fill=(235, 235, 235))
        draw.text((x0 - 74, yy - 10), f"{yv:.2f}", fill=(90, 90, 90), font=F_TICK)
    for i in range(n):
        xx = x0 + w * i / max(n - 1, 1)
        draw.text((xx - 5, y0 + h + 8), str(i + 1), fill=(90, 90, 90), font=F_TICK)
    draw.text((x0 + w / 2 - 18, y0 + h + 32), "step", fill=(60, 60, 60), font=F_AXIS)
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
            draw.line(pts, fill=col, width=3)
    if vmin < 0.5 < vmax:
        yz = y0 + h - h * (0.5 - vmin) / (vmax - vmin)
        draw.line([x0, yz, x0 + w, yz], fill=(200, 200, 200), width=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../product")
    ap.add_argument("--out", default="ablations/groups/G9_level_controller.png")
    args = ap.parse_args()
    root = os.path.join(args.root, "results", "ablation")
    gamma_series, shat_series = [], []
    for tag in SERIES:
        run = find_run(root, tag)
        if not run:
            print("missing", tag)
            continue
        cfg, gam, gap, steps = load(run)
        n = steps or len(gam)
        gam_ds = resample(gam, n)
        gap_ds = resample(gap, n)
        gamma_series.append((tag, gam_ds))
        if cfg.get("gamma_mode") == "ema_level":
            shat_series.append((tag, level_shat(gap_ds, cfg.get("rho", 0.5))))
        else:
            shat_series.append((tag, gap_ds))
    color_of = lambda tag: COLORS[SERIES.index(tag) % len(COLORS)]
    img = Image.new("RGB", (1960, 1100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 22), "controller over training: gamma and EMA/residual input", fill=(0, 0, 0), font=F_TITLE)
    lx, ly = 30, 78
    for tag in SERIES:
        draw.line([lx, ly + 9, lx + 34, ly + 9], fill=color_of(tag), width=5)
        draw.text((lx + 42, ly), tag, fill=(0, 0, 0), font=F_LEG)
        lx += 60 + int(draw.textlength(tag, font=F_LEG))
    panel(draw, 110, 150, 1760, 380, "gamma vs step  (gray line = 0.5)", gamma_series, color_of, vmin=0.0, vmax=1.0)
    panel(draw, 110, 650, 1760, 380, "controller input (ema_level: bias-corrected s_hat; others: gap / residual M)", shat_series, color_of)
    out = os.path.join(args.root, "record", args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    img.save(out)
    print("written", out)


if __name__ == "__main__":
    main()
