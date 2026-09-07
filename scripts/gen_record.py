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
           " & IT & PT & IT & PT & IT & PT \\\\")


def agg_cells(run_entry):
    data = parse_final(run_entry[3])
    cells = []
    for group in ("Forget", "Retain", "Real"):
        for task in ("Fill", "Classif", "Gen"):
            cells.append(scope_cell(data, group, task, "All"))
    return " & ".join(cells)


def pm_cells(run_entry):
    data = parse_final(run_entry[3])
    cells = []
    for group in ("Forget", "Retain", "Real"):
        for task in ("Fill", "Classif", "Gen"):
            for modal in ("IT", "PT"):
                cells.append(scope_cell(data, group, task, modal))
    return " & ".join(cells)


def metric_table(run_rows, header, cells_fn, ncols):
    lines = [
        "\\begin{table}[H]", "\\centering", "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "c" * ncols + "}", "\\toprule",
        header, "\\midrule",
    ]
    for run in run_rows:
        name = run[0]
        lines.append(f"\\textbf{{{esc(name)}}} & {cells_fn(run)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    return "\n".join(lines)


def hyper_table(entries):
    def scalar(v):
        return v is not None and not isinstance(v, (dict, list))

    cfgs = [e[2] for e in entries]
    cols = []
    for c in cfgs:
        for k, v in c.items():
            if k not in cols and scalar(v) and all(scalar(cc.get(k)) for cc in cfgs):
                cols.append(k)
    if not cols:
        cols = ["(no scalar hyperparameters)"]
    body = []
    for e in entries:
        cfg = e[2]
        row = [f"\\textbf{{{esc(e[0])}}}"]
        for k in cols:
            row.append(esc(cfg.get(k, "")))
        body.append(" & ".join(row) + " \\\\")
    head = "Run & " + " & ".join(esc(c) for c in cols) + " \\\\"
    ncols = len(cols) + 1
    return (f"\\begin{{table}}[H]\n\\centering\n\\resizebox{{\\linewidth}}{{!}}{{%\n"
            f"\\begin{{tabular}}{{l{'c'*(ncols-1)}}}\n\\toprule\n{head}\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n}\n\\end{table}")


def label_section(label, entries):
    out = [f"\\section{{{label}}}", f"\\label{{sec:experiment-{label.lower()}}}",
           "", "\\subsection{Hyperparameters}", "", hyper_table(entries), "",
           "\\subsection{Aggregate (All) scores}", "",
           metric_table(entries, HEAD_AGG, agg_cells, 9),
           "\\noindent\\small\\emph{Metric definitions: Aggregate = mean of IT and PT; "
           "forget lower is better, retain/real higher is better.}",
           "", "\\subsection{Per-modal (IT / PT) scores}", "",
           metric_table(entries, HEAD_PM, pm_cells, 18),
           "\\noindent\\small\\emph{Per-modal columns: IT = image-textual; PT = pure-text.}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../product")
    ap.add_argument("--out", default="UMU-Bench_实验记录.tex")
    ap.add_argument("--labels", default="", help="逗号分隔; 空=全部")
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
        "\\usepackage{multirow}", "\\usepackage{graphicx}",
        "\\usepackage{float}", "\\usepackage[margin=1in]{geometry}",
        "\\begin{document}", "\\title{UMU-Bench 实验记录}", "\\maketitle",
        "\\section*{Overview}",
        "\\subsection*{Aggregate (All) scores}",
        "",
    ]
    ov = build_overview(results_root)
    # Overview: 每 label 一行(行名=label), 仅收录最新 run 有有效 metrics 的 label
    ov_rows = []
    for label, entry in ov.items():
        if label not in labels or parse_final(entry[3]) is None:
            continue
        ov_rows.append((label,) + entry[1:])
    method_head = lambda h: h.replace("Run}", "Method}")
    doc.append(metric_table(ov_rows, method_head(HEAD_AGG), agg_cells, 9))
    doc.append("")
    doc.append("\\subsection*{Per-modal (IT / PT) scores}")
    doc.append("")
    doc.append(metric_table(ov_rows, method_head(HEAD_PM), pm_cells, 18))
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
