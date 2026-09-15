#!/usr/bin/env python3
"""E6 汇总: 读 results/_analysis/e6_runs.tsv, 输出每个 (alpha, epoch) 的
IT/PT 准确率、实体级 meanΔ、训练 margin 均值, 供 margin<->accuracy 标定。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_runs_tsv(path: Path):
    rows = []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split("\t")
    for line in lines[1:]:
        rows.append(dict(zip(header, line.split("\t"))))
    return rows


def task_accuracy(metrics_dir: Path, epoch_name: str):
    path = metrics_dir / f"{epoch_name}_final_evaluation_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    forget = data.get("Forget Set Results", {})
    fill = forget.get("fill_in_the_blank", {})
    qa = forget.get("QA (MM_QA/UM_QA) Results", {})
    return {
        "fill_it": fill.get("image_textual_accuracy"),
        "fill_pt": fill.get("pure_text_accuracy"),
        "qa_it": qa.get("Image-Textual Question Accuracy"),
        "qa_pt": qa.get("Pure Text Question Accuracy"),
    }


def last_margins(run_dir: Path, frac=0.25):
    path = run_dir / "logs" / "grad_norms.jsonl"
    if not path.exists():
        return None
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not recs:
        return None
    tail = recs[-max(1, int(len(recs) * frac)):]
    mm = [r["ratio_mm"] for r in tail if r.get("ratio_mm") is not None]
    um = [r["ratio_um"] for r in tail if r.get("ratio_um") is not None]
    if not mm or not um:
        return None
    return {"margin_mm": float(np.mean(mm)), "margin_um": float(np.mean(um)),
            "gap_um_minus_mm": float(np.mean(um) - np.mean(mm)),
            "n_steps": len(recs)}


def main():
    parser = argparse.ArgumentParser(description="E6 alpha 扫描汇总")
    parser.add_argument("--runs_tsv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs = read_runs_tsv(Path(args.runs_tsv))
    out = {"runs": []}
    for run in runs:
        run_dir = Path(run["run_dir"])
        entry = {"run_dir": str(run_dir), "alpha": float(run["alpha"]),
                 "modality": run.get("modality", "both"),
                 "oracle_dir": run["oracle_dir"], "epochs": []}
        for epoch_dir in sorted(run_dir.glob("runs/epoch-*")):
            if not (epoch_dir / "model").is_dir():
                continue
            epoch_name = epoch_dir.name
            metrics_dir = epoch_dir / "metrics"
            acc = task_accuracy(metrics_dir, epoch_name)
            stats_path = run_dir / "diagnosis" / f"{epoch_name}_stats.json"
            stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
            tasks = stats.get("tasks", {})
            qa_stats = tasks.get("qa", {})
            fill_stats = tasks.get("fill", {})
            entry["epochs"].append({
                "epoch": epoch_name,
                "qa_it_acc": acc.get("qa_it"),
                "qa_pt_acc": acc.get("qa_pt"),
                "qa_acc_gap": (acc["qa_it"] - acc["qa_pt"]) if acc.get("qa_it") is not None and acc.get("qa_pt") is not None else None,
                "qa_mean_delta": qa_stats.get("mean"),
                "qa_mean_delta_ci": qa_stats.get("mean_ci"),
                "qa_perm_p": qa_stats.get("permutation", {}).get("p"),
                "qa_tost_p": qa_stats.get("tost", {}).get("p"),
                "it_fill_acc": acc.get("fill_it"),
                "pt_fill_acc": acc.get("fill_pt"),
                "fill_mean_delta": fill_stats.get("mean"),
                "fill_perm_p": fill_stats.get("permutation", {}).get("p"),
            })
        entry["final_margins"] = last_margins(run_dir)
        out["runs"].append(entry)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'modality':>8} {'alpha':>8} {'epoch':>8} {'QA_IT':>7} {'QA_PT':>7} "
          f"{'QA_gap':>7} {'QA_meanD':>9} {'QA_p':>6} {'F_IT':>6} {'F_PT':>6} {'gap_um-mm':>10}")
    for run in out["runs"]:
        gm = run.get("final_margins") or {}
        for ep in run["epochs"]:
            def fmt(v, nd=3):
                return "NA" if v is None else f"{v:.{nd}f}"
            print(f"{run['modality']:>8} {run['alpha']:>8.4f} {ep['epoch']:>8} "
                  f"{fmt(ep['qa_it_acc'],1):>7} {fmt(ep['qa_pt_acc'],1):>7} "
                  f"{fmt(ep['qa_acc_gap'],1):>7} {fmt(ep['qa_mean_delta'],4):>9} "
                  f"{fmt(ep['qa_perm_p'],4):>6} {fmt(ep['it_fill_acc'],1):>6} "
                  f"{fmt(ep['pt_fill_acc'],1):>6} {fmt(gm.get('gap_um_minus_mm'),4):>10}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
