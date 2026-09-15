#!/usr/bin/env python3
"""E8 轴A: 实体级配对 Δ(e) 的统计推断。

主端点 (fill/classification): 同一实体同一题的 IT/PT 命中率差,
    Δ(e) = F_v(e) - F_t(e), F_m(e) = h0_m(e) - h_m(e)
条件化 (联合条件): 只保留 oracle 两侧都答对的题, 此时 h0_v=h0_t=1, Δ(e)=h_t(e)-h_v(e)。
检验: cluster bootstrap CI / 题级符号置换 / TOST 等价检验 (BH-FDR 跨任务校正)。
输入: oracle 与 unlearned 的 eval_vllm --dump_details 逐题明细。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TASKS = ("fill", "classification", "generation")
DETAIL_FILES = {
    "fill": ("fill_details.json", "Fill_Questions"),
    "classification": ("classification_details.json", "Classification_Questions"),
    "generation": ("generation_results.json", "Generation_Questions"),
}


def load_details(directory: Path, task: str):
    name, key = DETAIL_FILES[task]
    path = directory / f"forget_{name}"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[key]


def index_questions(records):
    return {(r["image_id"], r["question type"], r["qid"]): r for r in records}


def paired_records(oracle_records, unlearned_records):
    """{(image_id, qid): (o_it, o_pt, u_it, u_pt)}，要求 IT/PT 两侧都存在。"""
    o = index_questions(oracle_records)
    u = index_questions(unlearned_records)
    pairs = {}
    for (img, kind, qid), rec in o.items():
        if kind != "Image_Textual":
            continue
        key_pt = (img, "Pure_Text", qid)
        if key_pt not in o or (img, kind, qid) not in u or key_pt not in u:
            continue
        pairs[(img, qid)] = (
            rec, o[key_pt], u[(img, kind, qid)], u[key_pt],
        )
    return pairs


def binary_delta(o_it, o_pt, u_it, u_pt, joint_condition):
    if joint_condition:
        if not (o_it and o_pt):
            return None
        return float(u_pt) - float(u_it)
    return float(o_it - u_it) - float(o_pt - u_pt)


def entity_table(pairs, value_fn, joint_condition=True):
    by_entity = {}
    for (img, _), recs in pairs.items():
        d = value_fn(*recs)
        if d is None:
            continue
        by_entity.setdefault(img, []).append(d)
    return [(img, float(np.mean(ds)), len(ds), ds)
            for img, ds in sorted(by_entity.items())]


def bootstrap_summary(deltas, n_boot, rng):
    n = len(deltas)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = deltas[idx].mean(axis=1)
    return {
        "mean": float(deltas.mean()),
        "mean_ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "median": float(np.median(deltas)),
        "sd": float(deltas.std(ddof=1)) if n > 1 else 0.0,
        "iqr": [float(np.percentile(deltas, 25)), float(np.percentile(deltas, 75))],
        "p90_abs": float(np.percentile(np.abs(deltas), 90)),
        "max_abs": float(np.abs(deltas).max()),
        "c_stat": float(1.0 - np.abs(deltas).mean()),
        "n_entities": n,
    }


def permutation_pvalue(per_entity, n_boot, rng):
    all_d, sizes = [], []
    for _, _, n, ds in per_entity:
        all_d.extend(ds)
        sizes.append(n)
    d = np.asarray(all_d, dtype=np.float64)
    if d.size == 0:
        return None, None
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, d.size))
    flipped = signs * d
    entity_means = []
    start = 0
    for n in sizes:
        entity_means.append(flipped[:, start:start + n].mean(axis=1))
        start += n
    null = np.mean(np.stack(entity_means, axis=1), axis=1)
    observed = float(np.mean([m for _, m, _, _ in per_entity]))
    p = float((np.abs(null) >= abs(observed)).mean())
    return observed, p


def tost_pvalue(deltas, delta, n_boot, rng):
    n = len(deltas)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = deltas[idx].mean(axis=1)
    p = max(float((means <= -delta).mean()), float((means >= delta).mean()))
    return p, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test(per_entity):
    pos = sum(1 for _, m, _, _ in per_entity if m > 0)
    neg = sum(1 for _, m, _, _ in per_entity if m < 0)
    n = pos + neg
    if n == 0:
        return None
    k = min(pos, neg)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"n_pos": pos, "n_neg": neg, "p_two_sided": min(1.0, p)}


def bh_fdr(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        val = min(prev, pvals[idx] * m / rank)
        adjusted[idx] = val
        prev = val
    return adjusted.tolist()


def analyze_task(oracle_dir, unlearned_dir, task, delta, n_boot, seed):
    o = load_details(Path(oracle_dir), task)
    u = load_details(Path(unlearned_dir), task)
    pairs = paired_records(o, u)
    if not pairs:
        return {"error": "no paired IT/PT questions"}
    if task == "generation":
        def value_fn(o_it, o_pt, u_it, u_pt):
            return (float(o_it["rougeL"]) - float(u_it["rougeL"])) - \
                   (float(o_pt["rougeL"]) - float(u_pt["rougeL"]))
        per_entity = entity_table(pairs, value_fn, joint_condition=False)
        result = {"metric": "rougeL_difference"}
    else:
        def value_fn(o_it, o_pt, u_it, u_pt):
            return binary_delta(bool(o_it["correct"]), bool(o_pt["correct"]),
                                bool(u_it["correct"]), bool(u_pt["correct"]),
                                joint_condition=True)
        per_entity = entity_table(pairs, value_fn, joint_condition=True)
        result = {"metric": "hit_rate_difference_joint_condition"}
    if not per_entity:
        return {"error": "no entities after conditioning"}

    deltas = np.array([m for _, m, _, _ in per_entity])
    weights = np.array([n for _, _, n, _ in per_entity])
    rng = np.random.default_rng(seed)
    result.update(bootstrap_summary(deltas, n_boot, rng))
    result["n_questions"] = int(weights.sum())
    observed, p_perm = permutation_pvalue(per_entity, n_boot, rng)
    result["permutation"] = {"observed": observed, "p": p_perm}
    p_tost, lo, hi = tost_pvalue(deltas, delta, n_boot, rng)
    result["tost"] = {"delta": delta, "p": p_tost, "ci": [lo, hi],
                      "equivalent": bool(p_tost < 0.05 and lo > -delta and hi < delta)}
    result["sign_test"] = sign_test(per_entity)
    result["per_entity"] = [
        {"image_id": img, "delta": m, "n_questions": n} for img, m, n, _ in per_entity
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description="E8 轴A: 实体级配对 Δ(e) 统计推断")
    parser.add_argument("--oracle_dir", required=True)
    parser.add_argument("--unlearned_dir", required=True)
    parser.add_argument("--label", default="unlearned")
    parser.add_argument("--tasks", default="fill,classification")
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = {"oracle_dir": args.oracle_dir, "unlearned_dir": args.unlearned_dir,
               "label": args.label, "delta": args.delta, "tasks": {}}
    perm_tasks, perm_ps, tost_tasks, tost_ps = [], [], [], []
    for task in args.tasks.split(","):
        task = task.strip()
        if task not in TASKS:
            raise ValueError(f"unknown task: {task}")
        try:
            res = analyze_task(args.oracle_dir, args.unlearned_dir, task,
                               args.delta, args.bootstrap, args.seed)
        except FileNotFoundError as exc:
            res = {"error": f"missing details file: {exc}"}
        results["tasks"][task] = res
        if res.get("permutation", {}).get("p") is not None:
            perm_tasks.append(task)
            perm_ps.append(res["permutation"]["p"])
        if res.get("tost", {}).get("p") is not None:
            tost_tasks.append(task)
            tost_ps.append(res["tost"]["p"])
    if perm_ps:
        for t, p in zip(perm_tasks, bh_fdr(perm_ps)):
            results["tasks"][t]["permutation"]["p_fdr"] = p
    if tost_ps:
        for t, p in zip(tost_tasks, bh_fdr(tost_ps)):
            results["tasks"][t]["tost"]["p_fdr"] = p

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for task, res in results["tasks"].items():
        if "error" in res:
            print(f"[{task}] {res['error']}")
            continue
        print(f"[{task}] n_ent={res['n_entities']} n_q={res['n_questions']} "
              f"meanD={res['mean']:+.4f} CI={res['mean_ci']} "
              f"perm_p={res['permutation']['p']} tost_p={res['tost']['p']} "
              f"equiv={res['tost'].get('equivalent')}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
