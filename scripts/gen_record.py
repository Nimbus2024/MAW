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


def task_metric(scope, task, mode):
    """从单个 scope dict 提取 {IT,PT} 数值; 无则空串。mode IT/PT/ALL。"""
    td = scope.get(task) if isinstance(scope, dict) else None
    if not isinstance(td, dict):
        return ""
    if task == "fill_in_the_blank":
        it = td.get("image_textual_accuracy")
        pt = td.get("pure_text_accuracy")
    elif task == "classification":
        it = find_json_key(td, "image-textual question accuracy")
        pt = find_json_key(td, "pure text question accuracy")
    else:  # generation: use ROUGE-L (fallback ROUGE-1)
        it = find_json_key(td, "average rouge-l", "(image_textual)")
        pt = find_json_key(td, "average rouge-l", "(pure_text)")
        if it is None:
            it = find_json_key(td, "average rouge-1", "(image_textual)")
        if pt is None:
            pt = find_json_key(td, "average rouge-1", "(pure_text)")
    if it is None or pt is None:
        return ""
    if mode == "ALL":
        return fmt((it + pt) / 2.0)
    return fmt(it if mode == "IT" else pt)


def fmt(x):
    if isinstance(x, str):
        return x
    if abs(x - round(x)) < 1e-9:
        return f"{x:.0f}"
    if abs(x) < 2:
        return f"{x:.3f}"
    return f"{x:.1f}"


def parse_final(metrics_dir):
    files = glob.glob(os.path.join(metrics_dir, "*final_evaluation_results.json"))
    if not files:
        return None
    try:
        return json.load(open(files[0]))
    except Exception:
        return None


def scope_metric(run_data, group, task, mode):
    if not run_data:
        return ""
    for gkey in GROUP_KEYS[group]:
        scope = run_data.get(gkey)
        if isinstance(scope, dict):
            return task_metric(scope, TASK_KEYS[task], mode)
    return ""


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
            cells.append(scope_metric(data, group, task, "ALL"))
    return " & ".join(cells)


def pm_cells(run_entry):
    data = parse_final(run_entry[3])
    cells = []
    for group in ("Forget", "Retain", "Real"):
        for task in ("Fill", "Classif", "Gen"):
            for mode in ("IT", "PT"):
                cells.append(scope_metric(data, group, task, mode))
    return " & ".join(cells)


def metric_table(run_rows, header, cells_fn, ncols):
    lines = [
        "\\begin{table}[H]", "\\centering", "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l" + "c" * ncols + "}", "\\toprule",
        header, "\\midrule",
    ]
    for run in run_rows:
        name = run[0]
        lines.append(f"\\textbf{{{name}}} & {cells_fn(run)} \\\\")
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
        row = [f"\\textbf{{{e[0]}}}"]
        for k in cols:
            row.append(str(cfg.get(k, "")))
        body.append(" & ".join(row) + " \\\\")
    head = "Run & " + " & ".join(cols) + " \\\\"
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
        "\\usepackage{booktabs}", "\\usepackage{tabularx}",
        "\\usepackage{multirow}", "\\usepackage{graphicx}",
        "\\usepackage{float}", "\\usepackage[margin=1in]{geometry}",
        "\\begin{document}", "\\title{UMU-Bench 实验记录}", "\\maketitle",
        "\\section*{Overview}",
        "\\subsection*{Aggregate (All) scores}",
        "",
    ]
    ov = build_overview(results_root)
    ov_rows = [r for l, r in ov.items() if l in labels]
    doc.append(metric_table(ov_rows, HEAD_AGG, agg_cells, 9))
    doc.append("")
    doc.append("\\subsection*{Per-modal (IT / PT) scores}")
    doc.append("")
    doc.append(metric_table(ov_rows, HEAD_PM, pm_cells, 18))
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
