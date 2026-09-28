#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_record as G

HEAD = ("Method & \\multicolumn{3}{c}{Forget}"
        " & \\multicolumn{3}{c}{Retain} & \\multicolumn{3}{c}{Real} \\\\\n"
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n"
        " & Fill($\\downarrow$) & Classif($\\downarrow$) & Gen($\\downarrow$)"
        " & Fill($\\uparrow$) & Classif($\\uparrow$) & Gen($\\uparrow$)"
        " & Fill($\\uparrow$) & Classif($\\uparrow$) & Gen($\\uparrow$) \\\\")


def ratio_rows(results_root):
    if not os.path.isdir(results_root):
        return None
    ov = G.build_overview(results_root)
    labels = [l for l in ov if G.parse_final(ov[l][3]) is not None]
    methods = [l for l in sorted(labels) if l not in G.REF_LABELS]
    refs = [l for l in sorted(labels) if l in G.REF_LABELS]
    return [(l, ov[l]) for l in methods + refs]


def best_worst(entries):
    best, worst = {}, {}
    for ci, (g, t, m) in enumerate(G.AGG_COLSPEC):
        nums = [(e, G._cell_value(e, g, t, m)) for _, e in entries]
        nums = [(e, v) for e, v in nums if v is not None]
        if not nums:
            continue
        high = g != "Forget"
        best[ci] = (max if high else min)(nums, key=lambda x: x[1])[0]
        worst[ci] = (min if high else max)(nums, key=lambda x: x[1])[0]
    return best, worst


def cell_text(entry, ci, best, worst):
    g, t, m = G.AGG_COLSPEC[ci]
    v = G._cell_value(entry, g, t, m)
    s = "—" if v is None else G._fmt_val(t, v)
    if v is not None:
        if best.get(ci) is entry:
            s = f"\\textcolor{{umugreen}}{{{s}}}"
        elif worst.get(ci) is entry:
            s = f"\\textcolor{{umured}}{{{s}}}"
    return s


def row_line(label, entry, best, worst):
    cells = [f"\\textbf{{{G.esc(label)}}}"]
    cells += [cell_text(entry, ci, best, worst) for ci in range(len(G.AGG_COLSPEC))]
    return " & ".join(cells) + " \\\\"


def build_doc(root, ratios):
    ncols = len(G.AGG_COLSPEC) + 1
    lines = [
        "\\documentclass{article}",
        "\\usepackage[UTF8]{ctex}",
        "\\usepackage{booktabs}",
        "\\usepackage{longtable}",
        "\\usepackage{float}",
        "\\usepackage[table]{xcolor}",
        "\\definecolor{umugreen}{HTML}{228B22}",
        "\\definecolor{umured}{HTML}{B22222}",
        "\\usepackage[margin=1in]{geometry}",
        "\\begin{document}",
        "\\title{UMU-Bench Overview across forget ratios}",
        "\\maketitle",
        "\\section*{Aggregate (All) scores}",
        "",
        "{\\footnotesize\\setlength{\\tabcolsep}{3pt}",
        "\\begin{longtable}{l" + "c" * len(G.AGG_COLSPEC) + "}",
        "\\toprule",
        HEAD,
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        HEAD,
        "\\midrule",
        "\\endhead",
    ]
    for r in ratios:
        rows = ratio_rows(os.path.join(root, "results", f"ratio{r}"))
        lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textbf{{ratio={r}}}}} \\\\")
        lines.append("\\midrule")
        if not rows:
            lines.append(f"\\multicolumn{{{ncols}}}{{c}}{{\\emph{{no data}}}} \\\\")
            continue
        methods = [(l, e) for l, e in rows if l not in G.REF_LABELS]
        refs = [(l, e) for l, e in rows if l in G.REF_LABELS]
        best, worst = best_worst(methods)
        for label, entry in methods:
            lines.append(row_line(label, entry, best, worst))
        if refs:
            lines.append("\\midrule")
            for label, entry in refs:
                lines.append(row_line(label, entry, {}, {}))
    lines += [
        "\\bottomrule",
        "\\end{longtable}",
        "}",
        "",
        G.METRIC_NOTE,
        "\\end{document}",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../product")
    ap.add_argument("--ratios", default="5,10,15")
    ap.add_argument("--out", default="Overview_ratios.tex")
    args = ap.parse_args()
    ratios = [int(x) for x in args.ratios.split(",") if x.strip()]
    out_path = os.path.join(args.root, "record", args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(build_doc(args.root, ratios))
    print(f"written {out_path}")


if __name__ == "__main__":
    main()
