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

CFG_COLS = (("label", "method"), ("lr", "lr"), ("beta", "$\\beta$"), ("alpha", "$\\alpha$"),
            ("rho", "$\\rho$"), ("gamma_mode", "$\\gamma$mode"),
            ("gamma_fixed", "$\\gamma$fix"), ("coeff", "coeff"),
            ("max_steps", "max steps"))


def read_cfg(run):
    p = os.path.join(run, "config", "args.json")
    if os.path.isfile(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    return {}


def steps_per_epoch(run, cfg):
    log = os.path.join(run, "logs", "stdout.log")
    n = None
    if os.path.isfile(log):
        with open(log, encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.search(r"Forget(?: DPO)? pairs:\s*(\d+)", line)
                if m:
                    n = int(m.group(1))
                    break
    bs = cfg.get("batch_size") or cfg.get("global_batch_size") or 1
    if not n:
        return None
    return math.ceil(n / max(int(bs), 1))


def actual_steps(run, cfg):
    eps = sorted(glob.glob(os.path.join(run, "runs", "epoch-*")))
    spe = steps_per_epoch(run, cfg)
    if spe is None:
        return None
    done = len(eps) * spe
    ms = cfg.get("max_steps")
    return min(ms, done) if ms else done


def latest_metrics(run):
    eps = sorted(glob.glob(os.path.join(run, "runs", "epoch-*")),
                 key=lambda p: int(re.search(r"epoch-(\d+)", p).group(1)))
    if not eps:
        return None
    js = glob.glob(os.path.join(eps[-1], "metrics", "*final_evaluation_results.json"))
    return G.parse_final(os.path.dirname(js[0])) if js else None


def run_tag(run):
    m = re.fullmatch(r"\d{8}_\d{6}-(.+)", os.path.basename(run))
    return m.group(1) if m else os.path.basename(run)


def collect(results_root, epoch=None):
    rows = []
    labels = [d for d in sorted(os.listdir(results_root))
              if os.path.isdir(os.path.join(results_root, d))]
    for label in labels:
        for run in sorted(glob.glob(os.path.join(results_root, label, "*"))):
            if not os.path.isdir(run):
                continue
            if not re.fullmatch(r"\d{8}_\d{6}(-[A-Za-z0-9._]+)?", os.path.basename(run)):
                continue
            cfg = read_cfg(run)
            cfg["label"] = label
            data = latest_metrics(run)
            if data is None:
                continue
            rows.append((run_tag(run), cfg, data, actual_steps(run, cfg)))
    return rows


def best_worst(rows):
    best, worst = {}, {}
    for ci, (g, t, m) in enumerate(G.AGG_COLSPEC):
        pairs = []
        for tag, cfg, data, st in rows:
            it, pt, allv = G._task_vals(data, g, t)
            v = {"IT": it, "PT": pt, "All": allv}[m]
            pairs.append((tag, v))
        pairs = [(a, v) for a, v in pairs if v is not None]
        if not pairs:
            continue
        high = g != "Forget"
        key = (max if high else min)
        best[ci] = key(pairs, key=lambda x: x[1])[0]
        key2 = (min if high else max)
        worst[ci] = key2(pairs, key=lambda x: x[1])[0]
    return best, worst


def metric_cell(data, ci, row_tag, best, worst):
    g, t, m = G.AGG_COLSPEC[ci]
    it, pt, allv = G._task_vals(data, g, t)
    v = {"IT": it, "PT": pt, "All": allv}[m]
    s = "—" if v is None else G._fmt_val(t, v)
    if v is not None:
        if best.get(ci) == row_tag:
            s = f"\\textcolor{{umugreen}}{{{s}}}"
        elif worst.get(ci) == row_tag:
            s = f"\\textcolor{{umured}}{{{s}}}"
    return s


def build_doc(rows, epoch_note):
    head = "Tag & " + " & ".join(lbl for _, lbl in CFG_COLS) + " & steps"
    head += " & \\multicolumn{3}{c}{Forget} & \\multicolumn{3}{c}{Retain} & \\multicolumn{3}{c}{Real} \\\\\n"
    head += "\\cmidrule(lr){%d-%d} \\cmidrule(lr){%d-%d} \\cmidrule(lr){%d-%d}\n" % (
        2 + len(CFG_COLS) + 1, 4 + len(CFG_COLS) + 1,
        5 + len(CFG_COLS) + 1, 7 + len(CFG_COLS) + 1,
        8 + len(CFG_COLS) + 1, 10 + len(CFG_COLS) + 1)
    best, worst = best_worst(rows)
    lines = [
        "\\documentclass{article}",
        "\\usepackage[UTF8]{ctex}",
        "\\usepackage{booktabs}",
        "\\usepackage{longtable}",
        "\\usepackage[table]{xcolor}",
        "\\definecolor{umugreen}{HTML}{228B22}",
        "\\definecolor{umured}{HTML}{B22222}",
        "\\usepackage[margin=0.8in]{geometry}",
        "\\begin{document}",
        "\\title{MAW Ablation (forget ratio 5, $\\lambda=0$)}",
        "\\maketitle",
        "\\noindent\\small\\emph{%s}" % G.esc(epoch_note),
        "",
        "{\\scriptsize\\setlength{\\tabcolsep}{2pt}",
        "\\begin{longtable}{l" + "c" * (len(CFG_COLS) + 1 + 9) + "}",
        "\\toprule",
        head,
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        head,
        "\\midrule",
        "\\endhead",
    ]
    for tag, cfg, data, steps in rows:
        cells = [G.esc(tag)]
        for key, _ in CFG_COLS:
            v = cfg.get(key)
            cells.append("" if v is None else G.esc(v))
        cells.append("" if steps is None else str(steps))
        for ci in range(len(G.AGG_COLSPEC)):
            cells.append(metric_cell(data, ci, tag, best, worst))
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{longtable}",
        "}",
        "",
        G.METRIC_NOTE,
        "\\end{document}",
    ]
    return "\n".join(lines) + "\n"


PM_COLS = (("label", "method"), ("beta", "$\\beta$"), ("alpha", "$\\alpha$"),
           ("rho", "$\\rho$"), ("gamma_mode", "$\\gamma$mode"),
           ("gamma_fixed", "$\\gamma$fix"), ("coeff", "coeff"),
           ("max_steps", "max steps"))
PM_SPEC = [(g, t, m) for g in ("Forget", "Retain", "Real")
           for t in ("Fill", "Classif", "Gen") for m in ("IT", "PT")]


def pm_best_worst(rows):
    best, worst = {}, {}
    for ci, (g, t, m) in enumerate(PM_SPEC):
        pairs = []
        for tag, cfg, data, st in rows:
            it, pt, al = G._task_vals(data, g, t)
            v = {"IT": it, "PT": pt}[m]
            if v is not None:
                pairs.append((tag, v))
        if not pairs:
            continue
        high = g != "Forget"
        best[ci] = (max if high else min)(pairs, key=lambda x: x[1])[0]
        worst[ci] = (min if high else max)(pairs, key=lambda x: x[1])[0]
    return best, worst


def pm_cell(data, ci, tag, best, worst):
    g, t, m = PM_SPEC[ci]
    it, pt, al = G._task_vals(data, g, t)
    v = {"IT": it, "PT": pt}[m]
    s = "—" if v is None else G._fmt_val(t, v)
    if v is not None:
        if best.get(ci) == tag:
            s = f"\\textcolor{{umugreen}}{{{s}}}"
        elif worst.get(ci) == tag:
            s = f"\\textcolor{{umured}}{{{s}}}"
    return s


def pm_head(lead):
    start = lead + 1
    line1 = "Tag & " + " & ".join(lbl for _, lbl in PM_COLS) + " & steps"
    line1 += (" & \\multicolumn{6}{c}{Forget} & \\multicolumn{6}{c}{Retain}"
              " & \\multicolumn{6}{c}{Real} \\\\\n")
    line1 += (f"\\cmidrule(lr){{{start}-{start+5}}}"
              f" \\cmidrule(lr){{{start+6}-{start+11}}}"
              f" \\cmidrule(lr){{{start+12}-{start+17}}}\n")
    blanks = " & ".join([""] * lead)
    line2 = blanks + " & " + " & ".join(
        ["\\multicolumn{2}{c}{Fill}", "\\multicolumn{2}{c}{Classif}",
         "\\multicolumn{2}{c}{Gen}"] * 3) + " \\\\\n"
    cm = []
    for n in range(9):
        a = start + 2 * n
        cm.append(f"\\cmidrule(lr){{{a}-{a+1}}}")
    line2 += " ".join(cm) + "\n"
    line3 = blanks + " & " + " & ".join(["IT", "PT"] * 9) + " \\\\"
    return line1 + line2 + line3


def build_doc_pm(rows, note):
    lead = len(PM_COLS) + 2
    ncols = lead + len(PM_SPEC)
    best, worst = pm_best_worst(rows)
    lines = [
        "\\documentclass{article}",
        "\\usepackage[UTF8]{ctex}",
        "\\usepackage[landscape,margin=0.6in]{geometry}",
        "\\usepackage{booktabs}",
        "\\usepackage{longtable}",
        "\\usepackage[table]{xcolor}",
        "\\definecolor{umugreen}{HTML}{228B22}",
        "\\definecolor{umured}{HTML}{B22222}",
        "\\begin{document}",
        "\\title{MAW Ablation per-modal (forget ratio 5, $\\lambda=0$)}",
        "\\maketitle",
        "\\noindent\\small\\emph{%s}" % G.esc(note),
        "",
        "{\\scriptsize\\setlength{\\tabcolsep}{1.5pt}",
        "\\begin{longtable}{l" + "c" * (ncols - 1) + "}",
        "\\toprule",
        pm_head(lead),
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        pm_head(lead),
        "\\midrule",
        "\\endhead",
    ]
    for tag, cfg, data, st in rows:
        cells = [G.esc(tag)]
        for key, _ in PM_COLS:
            v = cfg.get(key)
            cells.append("" if v is None else G.esc(v))
        cells.append("" if st is None else str(st))
        cells += [pm_cell(data, ci, tag, best, worst) for ci in range(len(PM_SPEC))]
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{longtable}",
        "}",
        "",
        "\\noindent\\small\\emph{IT = image-textual, PT = pure-text; each cell = All-column equivalent for that modality.}",
        "\\end{document}",
    ]
    return "\n".join(lines) + "\n"



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../product")
    ap.add_argument("--sub", default="ablation")
    ap.add_argument("--out", default="ablations/MAW_ablation.tex")
    ap.add_argument("--note", default="metrics from last checkpoint of each run")
    ap.add_argument("--per-modal", action="store_true")
    ap.add_argument("--include-cal", action="store_true")
    args = ap.parse_args()
    results_root = os.path.join(args.root, "results", args.sub)
    rows = collect(results_root)
    if not rows:
        print(f"no runs under {results_root}")
        return
    if args.per_modal:
        if not args.include_cal:
            rows = [r for r in rows if not r[0].startswith("cal_")]
        out = args.out if args.out != "ablations/MAW_ablation.tex" else "ablations/MAW_ablation_pm.tex"
        doc = build_doc_pm(rows, args.note)
    else:
        out = args.out
        doc = build_doc(rows, args.note)
    out_path = os.path.join(args.root, "record", out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"written {out_path} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
