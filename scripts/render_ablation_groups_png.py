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
    "G10_retain_alpha": "G10  with retain loss: level-EMA alpha sweep (rho=0)",
    "G11_retain_rho": "G11  with retain loss: rho sweep (alpha=0.3)",
    "G6_level_vs_baseline": "G6  level-EMA vs residual-EMA vs fixed vs DPO",
    "G7_level_rho": "G7  level-EMA rho sweep (alpha=1)",
    "G8_level_alpha": "G8  level-EMA alpha sweep (rho=0.5)",
    "A0_budget": "A0  budget calibration (lr x steps)",
}
PANELS = [
    ("Forget Fill (All)  lower = more forgotten",
     lambda run, eps: [P.series_metric(run["epochs"][e], "Forget", "Fill") if e in run["epochs"] else None for e in eps]),
    ("Retain Fill (All)  higher = better kept",
     lambda run, eps: [P.series_metric(run["epochs"][e], "Retain", "Fill") if e in run["epochs"] else None for e in eps]),
    ("Real Fill (All)  higher = better",
     lambda run, eps: [P.series_metric(run["epochs"][e], "Real", "Fill") if e in run["epochs"] else None for e in eps]),
    ("Forget Fill IT - PT  (positive = IT less forgotten)",
     lambda run, eps: [(P.series_metric(run["epochs"][e], "Forget", "Fill", "IT")
                        - P.series_metric(run["epochs"][e], "Forget", "Fill", "PT"))
                       if e in run["epochs"] else None for e in eps]),
    ("Retain Fill IT - PT  (positive = IT better kept)",
     lambda run, eps: [(P.series_metric(run["epochs"][e], "Retain", "Fill", "IT")
                        - P.series_metric(run["epochs"][e], "Retain", "Fill", "PT"))
                       if e in run["epochs"] else None for e in eps]),
    ("gamma vs step  (gray = 0.5)",
     "gamma"),
]


def draw_panel(draw, x0, y0, w, h, title, series, color_of, xlabel="epoch", refline=None):
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(255, 255, 255), outline=(120, 120, 120), width=2)
    draw.text((x0, y0 - 30), title, fill=(0, 0, 0), font=F_PANEL)
    norm = []
    for item in series:
        tag = item[0]
        if len(item) == 3:
            norm.append((tag, item[1], item[2]))
        else:
            ys = item[1]
            norm.append((tag, list(range(1, len(ys) + 1)), ys))
    vals = [v for _, _, ys in norm for v in ys if v is not None]
    if not vals:
        return
    xmin = min(xs[0] for _, xs, _ in norm if xs)
    xmax = max(xs[-1] for _, xs, _ in norm if xs)
    vmin, vmax = min(vals), max(vals)
    pad = (vmax - vmin) * 0.12 or 1.0
    vmin -= pad
    vmax += pad
    for k in range(5):
        yv = vmin + (vmax - vmin) * k / 4
        yy = y0 + h - h * k / 4
        draw.line([x0, yy, x0 + w, yy], fill=(235, 235, 235))
        draw.text((x0 - 78, yy - 10), f"{yv:.2f}", fill=(90, 90, 90), font=F_TICK)
    if refline is not None and vmin <= refline <= vmax:
        yr = y0 + h - h * (refline - vmin) / (vmax - vmin)
        for xx in range(int(x0), int(x0 + w), 22):
            draw.line([xx, yr, xx + 11, yr], fill=(160, 160, 160), width=2)
        draw.text((x0 + w - 58, yr - 26), f"{refline:g}", fill=(120, 120, 120), font=F_TICK)
    for k in range(6):
        xv = xmin + (xmax - xmin) * k / 5
        xx = x0 + w * k / 5
        draw.text((xx - 10, y0 + h + 8), f"{int(xv)}", fill=(90, 90, 90), font=F_TICK)
    draw.text((x0 + w / 2 - 18, y0 + h + 32), xlabel, fill=(60, 60, 60), font=F_AXIS)
    for tag, xs, ys in norm:
        col = color_of(tag)
        pts = [(x0 + w * (x - xmin) / max(xmax - xmin, 1), y0 + h - h * (v - vmin) / (vmax - vmin))
               for x, v in zip(xs, ys) if v is not None]
        if len(pts) >= 2:
            draw.line(pts, fill=col, width=3)
        lastv = next((v for v in reversed(ys) if v is not None), None)
        if pts and lastv is not None:
            draw.text((pts[-1][0] + 6, pts[-1][1] - 10), f"{lastv:.1f}", fill=col, font=F_TICK)


def render_family(name, title, tags, runs, outdir):
    color_of = lambda tag: P.COLORS[tags.index(tag) % len(P.COLORS)]
    present = [t for t in tags if t in runs]
    img = Image.new("RGB", (1960, 360 + 3 * 530), (255, 255, 255))
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
    ep_all = sorted(set().union(*[set(runs[t]["epochs"]) for t in present]))
    for i, (label, kind) in enumerate(PANELS):
        x0 = 110 + (i % 2) * 900
        y0 = y_base + (i // 2) * 530
        series = []
        if kind == "gamma":
            for tag in present:
                gam = runs[tag]["gamma"]
                steps = runs[tag]["cfg"].get("max_steps") or max(len(gam) // 2, 1)
                xs = [round(j * (steps - 1) / max(len(gam) - 1, 1)) + 1 for j in range(len(gam))]
                series.append((tag, xs, gam))
            draw_panel(draw, x0, y0, 820, 460, label, series, color_of, xlabel="step", refline=0.5)
            continue
        for tag in present:
            ys = kind(runs[tag], ep_all)
            if any(v is not None for v in ys):
                series.append((tag, ys))
        draw_panel(draw, x0, y0, 820, 460, label, series, color_of, xlabel="epoch")
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
