#!/usr/bin/env python3
"""gen_record.py — 按 EXPERIMENT_RECORD_SPEC.md 从 product/results 生成 LaTeX 实验记录。

扫描 ROOT/product/results/<label>/<timestamp>:
  - 读取 config/args.json (超参表)
  - 读取 metrics: 有 runs/<epoch>/metrics 则逐 epoch; 否则 <ts>/metrics
  - metrics 取 `<ts>` 内 `*_final_evaluation_results.json`(eval_vllm 嵌套 schema)

输出 ROOT/product/record/<series>.tex。仅用 stdlib。
用法:
  python scripts/gen_record.py [--root ../product] [--series UMU-Bench_实验记录] [--labels origin]
"""
import argparse
import glob
import json
import os
import re

GROUP_KEYS = {
    "Forget": ("Forget Set Results",),
    "Retain": ("Retain Set (shared dataset) Results", "Retain Set Results"),
    "Real": ("Retain Set (real person) Results", "Real Person Set Results"),
}
TASK_KEYS = {"Fill": "fill_in_the_blank", "Classif": "classification", "Gen": "generation"}


def find_json_key(obj, *needles):
    for needle in needles:
        for k, v in obj.items():
            if isinstance(v, (int, float)) and needle.lower() in k.lower():
                return float(v)
    return None


def _task_vals(data, group, task):
    """返回 (IT, PT, All)。All 公式与参考 gen_latex._all_metric 一致:
    forget = (it+pt+(100-err))/3; retain = (it+pt+acc_all)/3; real = acc_all;
    gen 的 All 直接读后端的 All Modal Average ROUGE-L。"""
    if not data:
        return None, None, None
    dk = {"Forget": "forget", "Retain": "retain", "Real": "real"}[group]
    scope = None
    for gkey in GROUP_KEYS[group]:
        s = data.get(gkey)
        if isinstance(s, dict):
            scope = s
            break
    t = scope.get(TASK_KEYS[task]) if isinstance(scope, dict) else None
    if not isinstance(t, dict):
        return None, None, None
    if task == "Fill":
        it = t.get("image_textual_accuracy")
        pt = t.get("pure_text_accuracy")
    elif task == "Classif":
        it = t.get("Image-Textual Question Accuracy")
        pt = t.get("Pure Text Question Accuracy")
    else:
        return (t.get("Average ROUGE-L (Image_Textual)"),
                t.get("Average ROUGE-L (Pure_Text)"),
                t.get("All Modal Average ROUGE-L"))
    if it is None or pt is None:
        return it, pt, None
    acc_all = t.get("All Modal Question Accuracy")
    err = t.get("All Modal Question Error")
    if dk == "real":
        all_v = acc_all
    elif dk == "forget" and err is not None:
        all_v = (it + pt + (100.0 - err)) / 3.0
    elif dk == "retain" and acc_all is not None:
        all_v = (it + pt + acc_all) / 3.0
    else:
        all_v = None
    return it, pt, all_v


def scope_cell(data, group, task, modal):
    it, pt, all_v = _task_vals(data, group, task)
    value = {"IT": it, "PT": pt, "All": all_v}[modal]
    if value is None:
        return ""
    pattern = "%.1f" if task != "Gen" else "%.3f"
    return pattern % value


def fmt(x):
    if isinstance(x, str):
        return x
    if abs(x - round(x)) < 1e-9:
        return f"{x:.0f}"
    if abs(x) < 2:
        return f"{x:.3f}"
    return f"{x:.1f}"


def esc(s):
    """LaTeX 转义: 时间戳/路径里的 _、#、% 等在文本模式需转义。"""
    return (str(s).replace("\\", r"\textbackslash{}")
            .replace("{", r"\{").replace("}", r"\}")
            .replace("_", r"\_").replace("&", r"\&")
            .replace("%", r"\%").replace("#", r"\#").replace("$", r"\$"))


def parse_final(metrics_dir):
    files = glob.glob(os.path.join(metrics_dir, "*final_evaluation_results.json"))
    if not files:
        return None
    try:
        return json.load(open(files[0]))
    except Exception:
        return None


def collect_runs(label_dir):
    """返回 run entries: (run_label, ts, config_dict, metrics_dir)。"""
    entries = []
    for ts in sorted(os.listdir(label_dir)):
        ts_dir = os.path.join(label_dir, ts)
        if not os.path.isdir(ts_dir) or not re.fullmatch(r"\d{8}_\d{6}", ts):
            continue
        cfg = {}
        cfg_file = os.path.join(ts_dir, "config", "args.json")
        if os.path.isfile(cfg_file):
            try:
                cfg = json.load(open(cfg_file))
            except Exception:
                cfg = {}
        runs_dir = os.path.join(ts_dir, "runs")
        if os.path.isdir(runs_dir):
            for epoch in sorted(os.listdir(runs_dir)):
                ep_dir = os.path.join(runs_dir, epoch)
                if not os.path.isdir(ep_dir):
                    continue
                m = re.fullmatch(r"epoch-(\d+)", epoch)
                run_name = f"{ts} (epoch {int(m.group(1))})" if m else f"{ts} ({epoch})"
                entries.append((run_name, ts_dir, cfg,
                                os.path.join(ep_dir, "metrics")))
        else:
            entries.append((ts, ts_dir, cfg, os.path.join(ts_dir, "metrics")))
    return entries


def build_overview(results_root):
    """最新 ts(及最新 epoch) 每 label 一行。返回 label -> run_entry。"""
    rows = {}
    for label in sorted(os.listdir(results_root)):
        ld = os.path.join(results_root, label)
        if not os.path.isdir(ld):
            continue
        runs = collect_runs(ld)
        if runs:
            rows[label] = runs[-1]
    return rows


HEAD_AGG = ("\\multirow{2}{*}{Run} & \\multicolumn{3}{c}{Forget}"
            " & \\multicolumn{3}{c}{Retain} & \\multicolumn{3}{c}{Real} \\\\\n"
            "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n"
            " & Fill($\\downarrow$) & Classif($\\downarrow$) & Gen($\\downarrow$)"
            " & Fill($\\uparrow$) & Classif($\\uparrow$) & Gen($\\uparrow$)"
            " & Fill($\\uparrow$) & Classif($\\uparrow$) & Gen($\\uparrow$) \\\\")
HEAD_PM = ("\\multirow{3}{*}{Run} & \\multicolumn{6}{c}{Forget}"
           " & \\multicolumn{6}{c}{Retain} & \\multicolumn{6}{c}{Real} \\\\\n"
           "\\cmidrule(lr){2-7} \\cmidrule(lr){8-13} \\cmidrule(lr){14-19}\n"
           " & \\multicolumn{2}{c}{Fill} & \\multicolumn{2}{c}{Classif}"
           " & \\multicolumn{2}{c}{Gen}"
           " & \\multicolumn{2}{c}{Fill} & \\multicolumn{2}{c}{Classif}"
           " & \\multicolumn{2}{c}{Gen}"
           " & \\multicolumn{2}{c}{Fill} & \\multicolumn{2}{c}{Classif}"
           " & \\multicolumn{2}{c}{Gen} \\\\\n"
           "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}"
           " \\cmidrule(lr){8-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-13}"
           " \\cmidrule(lr){14-15} \\cmidrule(lr){16-17} \\cmidrule(lr){18-19}\n"
           " & IT & PT & IT & PT & IT & PT & IT & PT & IT & PT"
           " & IT & PT & IT & PT & IT & PT & IT & PT \\\\")
HEAD_AGG_EP = ("Timestamp & Epoch"
               " & \\multicolumn{3}{c}{Forget}"
               " & \\multicolumn{3}{c}{Retain} & \\multicolumn{3}{c}{Real} \\\\\n"
               "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}\n"
               " & & Fill($\\downarrow$) & Classif($\\downarrow$) & Gen($\\downarrow$)"
               " & Fill($\\uparrow$) & Classif($\\uparrow$) & Gen($\\uparrow$)"
               " & Fill($\\uparrow$) & Classif($\\uparrow$) & Gen($\\uparrow$) \\\\")
HEAD_PM_EP = ("Timestamp & Epoch"
              " & \\multicolumn{6}{c}{Forget}"
              " & \\multicolumn{6}{c}{Retain} & \\multicolumn{6}{c}{Real} \\\\\n"
              "\\cmidrule(lr){3-8} \\cmidrule(lr){9-14} \\cmidrule(lr){15-20}\n"
              " & & \\multicolumn{2}{c}{Fill} & \\multicolumn{2}{c}{Classif}"
              " & \\multicolumn{2}{c}{Gen}"
              " & \\multicolumn{2}{c}{Fill} & \\multicolumn{2}{c}{Classif}"
              " & \\multicolumn{2}{c}{Gen}"
              " & \\multicolumn{2}{c}{Fill} & \\multicolumn{2}{c}{Classif}"
              " & \\multicolumn{2}{c}{Gen} \\\\\n"
              "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}"
              " \\cmidrule(lr){9-10} \\cmidrule(lr){11-12} \\cmidrule(lr){13-14}"
              " \\cmidrule(lr){15-16} \\cmidrule(lr){17-18} \\cmidrule(lr){19-20}\n"
               " & & IT & PT & IT & PT & IT & PT & IT & PT & IT & PT"
               " & IT & PT & IT & PT & IT & PT & IT & PT \\\\")

METRIC_NOTE = ("\\noindent\\small\\emph{Metric definitions: per-task All(aggregate) 列 = "
               "Fill/Classif: Forget 用 $(\\mathit{IT}+\\mathit{PT}+100-\\mathit{AllErr})/3$, "
               "Retain 用 $(\\mathit{IT}+\\mathit{PT}+\\mathit{AllAcc})/3$, Real 取 $\\mathit{AllAcc}$; "
               "其中 AllAcc/AllErr = All Modal Question Accuracy/Error(配对双模态均对/均错占比)。"
               "Gen 的 All 直接取后端 All Modal Average ROUGE-L"
               "(IT/PT 配对: Forget $(\\mathit{IT}^2+\\mathit{PT}^2)/(\\mathit{IT}+\\mathit{PT})$, "
               "Retain/Real $2\\,\\mathit{IT}\\,\\mathit{PT}/(\\mathit{IT}+\\mathit{PT})$)。"
               "Forget 越低越好, Retain/Real 越高越好.}")


def _epoch_of(entry):
    m = re.search(r"\(epoch (\d+)\)", entry[0])
    return m.group(1) if m else None


def metric_table(run_rows, header, colspec):
    """逐 run/epoch 指标表: longtable 跨页, 每列最优标绿/最差标红。"""
    groups = []
    for e in run_rows:
        ts = os.path.basename(e[1])
        if groups and groups[-1][0] == ts:
            groups[-1][1].append(e)
        else:
            groups.append([ts, [e]])
    rows = [e for _, es in groups for e in es]
    best, worst = {}, {}
    for ci, (g, t, m) in enumerate(colspec):
        nums = []
        for r in rows:
            v = _cell_value(r, g, t, m)
            if v is not None:
                nums.append((r, v))
        if not nums:
            continue
        high = g != "Forget"
        best[ci] = (max if high else min)(nums, key=lambda x: x[1])[0]
        worst[ci] = (min if high else max)(nums, key=lambda x: x[1])[0]
    body = []
    for ts, es in groups:
        for i, run in enumerate(es):
            ts_cell = f"\\textbf{{{esc(ts)}}}" if i == 0 else ""
            ep = _epoch_of(run) or ""
            cells = []
            for ci, (g, t, m) in enumerate(colspec):
                s = _fmt_val(t, _cell_value(run, g, t, m))
                if s:
                    if best.get(ci) is run:
                        s = f"\\textcolor{{umugreen}}{{{s}}}"
                    elif worst.get(ci) is run:
                        s = f"\\textcolor{{umured}}{{{s}}}"
                cells.append(s)
            body.append(f"{ts_cell} & {ep} & " + " & ".join(cells) + " \\\\")
    ncols = len(colspec)
    return ("{\\footnotesize\\setlength{\\tabcolsep}{1.5pt}\n"
            "\\begin{longtable}{ll" + "c" * ncols + "}\n"
            "\\toprule\n" + header + "\n\\midrule\n"
            "\\endfirsthead\n"
            "\\toprule\n" + header + "\n\\midrule\n"
            "\\endhead\n"
            + "\n".join(body) + "\n"
            "\\bottomrule\n\\end{longtable}\n}")


def hyper_table(entries):
    def scalar(v):
        return v is not None and not isinstance(v, (dict, list))

    by_ts = {}
    for e in entries:
        by_ts.setdefault(os.path.basename(e[1]), e)
    run_entries = [by_ts[k] for k in sorted(by_ts)]
    cfgs = [e[2] for e in run_entries]
    cols = []
    for c in cfgs:
        for k, v in c.items():
            is_path = isinstance(v, str) and v.startswith("/")
            if k not in cols and scalar(v) and not is_path and \
                    all(scalar(cc.get(k)) and not (isinstance(cc.get(k), str)
                                                    and cc.get(k).startswith("/"))
                        for cc in cfgs):
                cols.append(k)
    if not cols:
        cols = ["(no scalar hyperparameters)"]
    body = []
    for e in run_entries:
        cfg = e[2]
        row = [f"\\textbf{{{esc(os.path.basename(e[1]))}}}"]
        for k in cols:
            row.append(esc(cfg.get(k, "")))
        body.append(" & ".join(row) + " \\\\")
    head = "Run & " + " & ".join(esc(c) for c in cols) + " \\\\"
    ncols = len(cols) + 1
    return (f"\\begin{{table}}[H]\n\\centering\n\\resizebox{{\\linewidth}}{{!}}{{%\n"
            f"\\begin{{tabular}}{{l{'c'*(ncols-1)}}}\n\\toprule\n{head}\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n}\n\\end{table}")


def label_section(label, entries):
    out = [f"\\section{{{label}}}", f"\\label{{sec:experiment-{label.lower()}}}"]
    # 无超参(如 vanilla/origin 纯评估)不生成 Hyperparameters 小节
    if any(bool({k: v for k, v in e[2].items()
                 if v is not None and not isinstance(v, (dict, list))}) for e in entries):
        out += ["", "\\subsection{Hyperparameters}", "", hyper_table(entries), ""]
    out += [
        "", "\\subsection{Aggregate (All) scores}", "",
        metric_table(entries, HEAD_AGG_EP, AGG_COLSPEC),
        METRIC_NOTE,
        "", "\\subsection{Per-modal (IT / PT) scores}", "",
        metric_table(entries, HEAD_PM_EP, PM_COLSPEC),
        "\\noindent\\small\\emph{Per-modal columns: IT = image-textual; PT = pure-text; "
        "All 列公式见上.}"]
    return "\n".join(out)


GROUPS = ("Forget", "Retain", "Real")
TASKS = ("Fill", "Classif", "Gen")
AGG_COLSPEC = [(g, t, "All") for g in GROUPS for t in TASKS]
PM_COLSPEC = [(g, t, m) for g in GROUPS for t in TASKS for m in ("IT", "PT")]


def _cell_value(entry, g, t, m):
    data = parse_final(entry[3])
    it, pt, allv = _task_vals(data, g, t)
    return {"IT": it, "PT": pt, "All": allv}[m]


def _fmt_val(t, v):
    if v is None:
        return ""
    return ("%.1f" if t != "Gen" else "%.3f") % v


def overview_table(rows, header, ncols, colspec):
    """Overview: 每列用更好方标绿(umugreen)、更差方标红(umured)。
    Forget 越低越好; Retain/Real 越高越好。"""
    best, worst = {}, {}
    for ci, (g, t, m) in enumerate(colspec):
        vals = [(r, _cell_value(r, g, t, m)) for r in rows]
        nums = [(r, v) for r, v in vals if v is not None]
        if not nums:
            continue
        high = g != "Forget"
        best[ci] = max(nums, key=lambda x: x[1] if high else -x[1])[0]
        worst[ci] = min(nums, key=lambda x: x[1] if high else -x[1])[0]
    lines = ["\\begin{table}[H]", "\\centering", "\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{l" + "c" * ncols + "}", "\\toprule",
             header, "\\midrule"]
    for ri, row in enumerate(rows):
        cells = [f"\\textbf{{{esc(row[0])}}}"]
        for ci, (g, t, m) in enumerate(colspec):
            v = _cell_value(row, g, t, m)
            s = _fmt_val(t, v)
            if s:
                if best.get(ci) is row:
                    s = f"\\textcolor{{umugreen}}{{{s}}}"
                elif worst.get(ci) is row:
                    s = f"\\textcolor{{umured}}{{{s}}}"
            cells.append(s)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../product")
    ap.add_argument("--out", default="UMU-Bench_实验记录.tex")
    ap.add_argument("--labels", default="", help="逗号分隔; 空=全部")
    ap.add_argument("--pick", default="",
                    help="Overview 每 label 选定的代表 run, 逗号分隔 label=ts[@epoch]; "
                         "如 MAW=20260907_154223@8")
    args = ap.parse_args()

    results_root = os.path.join(args.root, "results")
    record_dir = os.path.join(args.root, "record")
    labels = [l for l in (x.strip() for x in args.labels.split(",")) if l] or \
        sorted(os.listdir(results_root))
    labels = [l for l in labels if os.path.isdir(os.path.join(results_root, l))]

    doc = [
        "\\documentclass{article}",
        "\\usepackage[UTF8]{ctex}",   # 中文(标题/说明), 建议 xelatex 编译
        "\\usepackage{booktabs}", "\\usepackage{tabularx}",
        "\\usepackage{longtable}",
        "\\usepackage{multirow}", "\\usepackage{graphicx}",
        "\\usepackage{float}", "\\usepackage[table]{xcolor}",
        "\\definecolor{umugreen}{HTML}{228B22}", "\\definecolor{umured}{HTML}{B22222}",
        "\\usepackage[margin=1in]{geometry}",
        "\\begin{document}", "\\title{UMU-Bench 实验记录}", "\\maketitle",
        "\\section*{Overview}",
        "\\subsection*{Aggregate (All) scores}",
        "",
    ]
    ov = build_overview(results_root)

    def _find_entry(entries, ts, epoch):
        for e in entries:
            if e[0].startswith(ts):
                if epoch is None or f"(epoch {epoch})" in e[0]:
                    return e
        return None

    for part in (x.strip() for x in args.pick.split(",") if x.strip()):
        label, _, rest = part.partition("=")
        ts, _, ep = rest.partition("@")
        if not os.path.isdir(os.path.join(results_root, label)):
            continue
        hit = _find_entry(collect_runs(os.path.join(results_root, label)), ts,
                          int(ep) if ep else None)
        if hit is not None:
            ov[label] = hit
            print(f"pick: {label} -> {hit[0]}")
        else:
            print(f"pick: {label} 未匹配 {ts}@{ep}, 保留默认")
    # Overview: 每 label 一行(行名=label), 仅收录最新 run 有有效 metrics 的 label
    ov_rows = []
    for label, entry in ov.items():
        if label not in labels or parse_final(entry[3]) is None:
            continue
        ov_rows.append((label,) + entry[1:])
    method_head = lambda h: h.replace("Run}", "Method}")
    agg_cols = AGG_COLSPEC
    pm_cols = PM_COLSPEC
    doc.append(overview_table(ov_rows, method_head(HEAD_AGG), 9, agg_cols))
    doc.append("")
    doc.append(METRIC_NOTE)
    doc.append("")
    doc.append("\\subsection*{Per-modal (IT / PT) scores}")
    doc.append("")
    doc.append(overview_table(ov_rows, method_head(HEAD_PM), 18, pm_cols))
    doc.append("")
    doc.append("\\noindent\\small\\emph{Per-modal columns: IT = image-textual; PT = pure-text; "
               "green = best value, red = worst value per column.}")
    doc.append("")
    for label in labels:
        entries = collect_runs(os.path.join(results_root, label))
        if not entries:
            continue
        doc.append(label_section(label, entries))
    doc.append("\\end{document}")

    os.makedirs(record_dir, exist_ok=True)
    out_path = os.path.join(record_dir, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc) + "\n")
    print(f"written {out_path}")


if __name__ == "__main__":
    main()
