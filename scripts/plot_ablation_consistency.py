#!/usr/bin/env python3
"""P1/P2 consistency visualization for the MAW ablation (per references/think/P0-P4).

P1: a-d plane (x=d=AllAcc, y=a=AllErr) with iso-oe lines, 3x2 (scope x task).
P2: forget-retain / forget-real Pareto, 2x2 (task x pair), top-right = Pareto.

a/d/oe are taken directly from the aggregate `All Modal Question Accuracy/Error`
(= d/a) in `*final_evaluation_results.json`; oe = 100 - a - d.
"""
import argparse
import glob
import json
import os
from PIL import Image, ImageDraw, ImageFont

FORGET = "Forget Set Results"
RETAIN = "Retain Set (shared dataset) Results"
REAL = "Retain Set (real person) Results"
SCOPES = [("Forget", FORGET), ("Retain", RETAIN), ("Real", REAL)]
TASKS = [("Fill", "fill_in_the_blank"), ("Cls", "classification")]

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/dejavu/",
             os.path.expanduser("~/.fonts/")]


def _font(name, size):
    for d in FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_TITLE = _font("DejaVuSans-Bold.ttf", 34)
F_PANEL = _font("DejaVuSans-Bold.ttf", 24)
F_LEG = _font("DejaVuSans.ttf", 22)
F_TICK = _font("DejaVuSans.ttf", 18)
F_AXIS = _font("DejaVuSans-Bold.ttf", 20)

# tag, path glob, family, color, epoch (None=final), star?
RUNS = [
    ("origin", "ratio5/origin/20260906_204146", "origin", (218, 165, 32), 0, True),
    ("oldMAW 170751", "ratio5/MAW/20260907_170751", "lambda1.5", (139, 0, 0), 6, False),
    ("PhaseC a0.3", "ablation/MAW/*-emaRL_a0.3", "lambda1.5", (230, 90, 90), 6, False),
    ("v1ema", "ablation/MAW/*-v1ema", "dyn", (31, 119, 180), 4, False),
    ("v4it", "ablation/MAW/*-v4it", "iso", (44, 160, 44), 4, False),
    ("v5pt", "ablation/MAW/*-v5pt", "iso", (255, 127, 14), 4, False),
    ("v6a3", "ablation/MAW/*-v6a3", "alpha", (148, 103, 189), 4, False),
    ("v8b04", "ablation/MAW/*-v8b04", "beta", (140, 86, 75), 4, False),
    ("v9coeff1", "ablation/MAW/*-v9coeff1", "coeff", (227, 119, 194), 4, False),
    ("v10dpo", "ablation/DPO/*-v10dpo", "loss", (23, 190, 207), 4, False),
    ("v11npo", "ablation/NPO/*-v11npo", "loss", (170, 170, 30), 4, False),
]


def _final_json(run_dir, epoch):
    if epoch:
        c = sorted(glob.glob(f"{run_dir}/runs/epoch-{epoch}/metrics/*final_evaluation_results.json"))
        if not c:
            c = sorted(glob.glob(f"{run_dir}/runs/epoch-*/metrics/*final_evaluation_results.json"))
    else:
        c = (sorted(glob.glob(f"{run_dir}/runs/epoch-*/metrics/*final_evaluation_results.json"))
             or sorted(glob.glob(f"{run_dir}/metrics/*final_evaluation_results.json")))
    return c[-1] if c else None


def load_runs(root):
    out = []
    for tag, pat, family, color, epoch, star in RUNS:
        dirs = sorted(glob.glob(os.path.join(root, "results", pat)))
        if not dirs:
            print(f"  ! missing {tag} ({pat})")
            continue
        fj = _final_json(dirs[0], epoch)
        if not fj:
            print(f"  ! no final json {tag}")
            continue
        m = json.load(open(fj))
        data = {}
        for sname, skey in SCOPES:
            for tname, tkey in TASKS:
                try:
                    a = m[skey][tkey]["All Modal Question Error"]
                    d = m[skey][tkey]["All Modal Question Accuracy"]
                except KeyError:
                    continue
                data[(sname, tname)] = dict(a=a, d=d, oe=100.0 - a - d, ep=epoch or "final")
        ep = epoch or "final"
        out.append(dict(tag=tag, family=family, color=color, star=star, ep=ep, data=data))
    return out


def _m(panel, x0, y0, w, h, xmin=0, xmax=100, ymin=0, ymax=100):
    return lambda x, y: (x0 + w * (x - xmin) / (xmax - xmin), y0 + h - h * (y - ymin) / (ymax - ymin))


def draw_axes(draw, x0, y0, w, h, title, xlabel, ylabel):
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(255, 255, 255), outline=(90, 90, 90), width=2)
    draw.text((x0, y0 - 34), title, fill=(0, 0, 0), font=F_PANEL)
    P = _m(None, x0, y0, w, h)
    for v in range(0, 101, 20):
        xx, _ = P(v, 0)
        draw.line([xx, y0, xx, y0 + h], fill=(238, 238, 238))
        draw.text((xx - 12, y0 + h + 6), str(v), fill=(90, 90, 90), font=F_TICK)
        _, yy = P(0, v)
        draw.line([x0, yy, x0 + w, yy], fill=(238, 238, 238))
        draw.text((x0 - 42, yy - 10), str(v), fill=(90, 90, 90), font=F_TICK)
    draw.text((x0 + w / 2 - 30, y0 + h + 34), xlabel, fill=(60, 60, 60), font=F_AXIS)
    draw.text((x0 - 40, y0 - 6), ylabel, fill=(60, 60, 60), font=F_AXIS)


def dot(draw, cx, cy, color, star=False, r=8):
    if star:
        import math
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rad = r + 7 if i % 2 == 0 else (r + 7) * 0.45
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        draw.polygon(pts, fill=color, outline=(80, 60, 0))
    else:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(255, 255, 255), width=2)


def render_p1(runs, outdir):
    pw, ph = 560, 470
    cols, rows = 2, 3
    W, H = 60 + cols * (pw + 70), 260 + rows * (ph + 90)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((40, 24), "P1 (ablation): a-d plane with iso-oe lines  (x=d=AllAcc, y=a=AllErr)",
              fill=(0, 0, 0), font=F_TITLE)
    draw.text((40, 72), "Forget: better up-left (a up) | Retain/Real: better right-down (d up) | same d: higher a = lower oe",
              fill=(70, 70, 70), font=F_LEG)
    # legend
    lx, ly = 40, 120
    for r in runs:
        draw.line([lx, ly + 9, lx + 30, ly + 9], fill=r["color"], width=6)
        if r["star"]:
            dot(draw, lx + 14, ly + 9, r["color"], star=True, r=6)
        lbl = f"{r['tag']}"
        draw.text((lx + 38, ly), lbl, fill=(0, 0, 0), font=F_LEG)
        lx += 52 + int(draw.textlength(lbl, font=F_LEG))
        if lx > W - 260:
            lx, ly = 40, ly + 32
    for ri, (sname, skey) in enumerate(SCOPES):
        for ci, (tname, tkey) in enumerate(TASKS):
            x0 = 60 + ci * (pw + 70)
            y0 = 220 + ri * (ph + 90)
            draw_axes(draw, x0, y0, pw, ph, f"{sname} / {tname}", "d = AllAcc (both right)",
                      "a = AllErr (both wrong)")
            P = _m(None, x0, y0, pw, ph)
            oes = [r["data"][(sname, tname)]["oe"] for r in runs if (sname, tname) in r["data"]]
            if oes:
                lo, hi = max(0, int(min(oes)) - 4), min(100, int(max(oes)) + 4)
                for oe in range(lo, hi + 1, 5):
                    c = 100 - oe
                    if c <= 100:
                        p1, p2 = P(max(0, c - 100), min(100, c)), P(min(100, c), max(0, c - 100))
                    else:
                        p1, p2 = P(c - 100, 100), P(100, c - 100)
                    draw.line([p1, p2], fill=(226, 226, 226), width=1)
                    lx2, ly2 = P(c * 0.5, c * 0.5)
                    draw.text((lx2 + 4, ly2 + 2), f"oe={oe}", fill=(170, 170, 170), font=F_TICK)
            # oe = 0 boundary
            p1, p2 = P(0, 100), P(100, 0)
            draw.line([p1, p2], fill=(120, 120, 120), width=2)
            for r in runs:
                v = r["data"].get((sname, tname))
                if not v:
                    continue
                cx, cy = P(v["d"], v["a"])
                if r["star"]:
                    # bold iso-oe line for origin
                    c = v["a"] + v["d"]
                    if c <= 100:
                        q1, q2 = P(max(0, c - 100), min(100, c)), P(min(100, c), max(0, c - 100))
                    else:
                        q1, q2 = P(c - 100, 100), P(100, c - 100)
                    draw.line([q1, q2], fill=r["color"], width=2)
                dot(draw, cx, cy, r["color"], star=r["star"])
    path = os.path.join(outdir, "fig_p11_isooe_ablation.png")
    img.save(path)
    return path


def _pareto(pts):
    fr = []
    for p in pts:
        if not any(q[0] >= p[0] and q[1] >= p[1] and (q[0] > p[0] or q[1] > p[1]) for q in pts if q is not p):
            fr.append(p)
    return sorted(fr, key=lambda t: t[0])


def render_p2(runs, outdir):
    pw, ph = 560, 470
    cols, rows = 2, 2
    W, H = 60 + cols * (pw + 70), 260 + rows * (ph + 90)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((40, 24), "P2 (ablation): forget-retain / forget-real Pareto  (top-right = Pareto)",
              fill=(0, 0, 0), font=F_TITLE)
    draw.text((40, 72), "y = a_forget (both wrong, up better) | x = d (both right, right better) | red dashed = non-dominated frontier",
              fill=(70, 70, 70), font=F_LEG)
    lx, ly = 40, 120
    for r in runs:
        draw.line([lx, ly + 9, lx + 30, ly + 9], fill=r["color"], width=6)
        if r["star"]:
            dot(draw, lx + 14, ly + 9, r["color"], star=True, r=6)
        lbl = r["tag"]
        draw.text((lx + 38, ly), lbl, fill=(0, 0, 0), font=F_LEG)
        lx += 52 + int(draw.textlength(lbl, font=F_LEG))
        if lx > W - 260:
            lx, ly = 40, ly + 32
    pairs = [("f-ret", "Retain"), ("f-real", "Real")]
    for ri, (tname, tkey) in enumerate(TASKS):
        for ci, (pname, other) in enumerate(pairs):
            x0 = 60 + ci * (pw + 70)
            y0 = 220 + ri * (ph + 90)
            draw_axes(draw, x0, y0, pw, ph, f"{tname}: a_forget vs d_{other}",
                      f"d_{other} = AllAcc (both right)", "a_forget = AllErr (both wrong)")
            P = _m(None, x0, y0, pw, ph)
            pts = []
            for r in runs:
                vf = r["data"].get(("Forget", tname))
                vo = r["data"].get((other, tname))
                if not vf or not vo:
                    continue
                pts.append((vo["d"], vf["a"], r))
            # pareto frontier (larger x, larger y better)
            fr = _pareto([(p[0], p[1]) for p in pts])
            if len(fr) >= 2:
                for k in range(len(fr) - 1):
                    a1, a2 = P(*fr[k]), P(*fr[k + 1])
                    _dashed(draw, a1, a2, (214, 39, 40), 2)
                # extend to axes
                _dashed(draw, P(0, fr[0][1]), P(*fr[0]), (214, 39, 40), 2)
                _dashed(draw, P(*fr[-1]), P(fr[-1][0], 0), (214, 39, 40), 2)
            for x, y, r in pts:
                if r["star"]:
                    _dashed(draw, P(x, 0), P(x, y), r["color"], 1)
                    _dashed(draw, P(0, y), P(x, y), r["color"], 1)
                cx, cy = P(x, y)
                dot(draw, cx, cy, r["color"], star=r["star"])
    path = os.path.join(outdir, "fig_p12_pareto_ablation.png")
    img.save(path)
    return path


def _dashed(draw, p1, p2, color, width, dash=8, gap=6):
    import math
    x1, y1 = p1
    x2, y2 = p2
    d = math.hypot(x2 - x1, y2 - y1)
    if d == 0:
        return
    n = int(d // (dash + gap)) + 1
    for i in range(n):
        t0 = i * (dash + gap) / d
        t1 = min((i * (dash + gap) + dash) / d, 1.0)
        draw.line([(x1 + (x2 - x1) * t0, y1 + (y2 - y1) * t0),
                   (x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1)], fill=color, width=width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../product")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    runs = load_runs(a.root)
    outdir = a.outdir or os.path.join(a.root, "record", "ablations", "groups")
    os.makedirs(outdir, exist_ok=True)
    print("runs:", [r["tag"] for r in runs])
    print("written", render_p1(runs, outdir))
    print("written", render_p2(runs, outdir))
    # print table
    print("\nrun                  | forget a/d/oe (fill,cls) | retain a/d/oe | real a/d/oe")
    for r in runs:
        cells = []
        for sname, _ in SCOPES:
            vals = []
            for tname, _ in TASKS:
                v = r["data"].get((sname, tname))
                vals.append("--" if not v else f"{v['a']:.0f}/{v['d']:.0f}/{v['oe']:.0f}")
            cells.append(",".join(vals))
        print(f"{r['tag']:20s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
