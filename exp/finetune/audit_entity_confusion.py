"""Audit whether short factual errors copy another profile's attribute value."""

import argparse
import ast
import difflib
import json
import re

import pandas as pd


def normalize(value):
    value = re.sub(r"</?s_[^>]+>", " ", str(value))
    value = value.replace("<sep/>", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(value.split())


def load_lookup(parquet_path):
    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    qa_by_id = {}
    profile_answers = {}
    answers_by_key = {}
    for _, row in df.iterrows():
        pid = str(row.get("ID"))
        qa_by_id[pid] = {}
        profile_answers[pid] = {}
        for column in ("UM_QA", "MM_QA"):
            value = row.get(column)
            if isinstance(value, str):
                value = ast.literal_eval(value)
            if not isinstance(value, dict):
                continue
            questions = value.get("question", {}) or {}
            answers = value.get("answer", {}) or {}
            for key, question in questions.items():
                key = str(key)
                answer = str(answers.get(key, ""))
                qa_by_id[pid][str(question)] = (key, answer)
                profile_answers[pid].setdefault(key, answer)
                answers_by_key.setdefault(key, []).append((pid, answer))
    return qa_by_id, profile_answers, answers_by_key


def categorize(pid, key, prediction, profile_answers, answers_by_key):
    pred = normalize(prediction)
    if not pred:
        return "empty", []
    exact = []
    near = []
    other_attr = []
    for other_id, value in answers_by_key.get(key, []):
        if other_id == pid:
            continue
        value_norm = normalize(value)
        if not value_norm:
            continue
        if pred == value_norm:
            exact.append((other_id, value))
        elif len(pred) >= 3 and (pred in value_norm or value_norm in pred):
            near.append((other_id, value))
        elif difflib.SequenceMatcher(None, pred, value_norm).ratio() >= 0.85:
            near.append((other_id, value))
    for other_key, value in profile_answers.get(pid, {}).items():
        if other_key != key and pred == normalize(value):
            other_attr.append((other_key, value))
    if exact:
        return "other_entity_exact", exact[:3]
    if near:
        return "other_entity_near", near[:3]
    if other_attr:
        return "same_entity_other_attr", other_attr[:3]
    return "unexplained", []


def parse_args():
    parser = argparse.ArgumentParser(description="Audit entity-attribute confusion")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--details", required=True)
    parser.add_argument("--field", default="")
    parser.add_argument("--max_gold_len", type=int, default=80)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def load_details(path, field):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if field:
        return data[field]
    for key in ("train_details", "details", "results"):
        if isinstance(data.get(key), list):
            return data[key]
    details = []
    for value in data.values():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            details.extend(value)
    return details


def main():
    args = parse_args()
    qa_by_id, profile_answers, answers_by_key = load_lookup(args.parquet)
    details = load_details(args.details, args.field)
    counts = {}
    examples = []
    short_wrong = 0
    unmapped = 0
    for item in details:
        question = str(item.get("question", ""))
        gold = str(item.get("ground_truth", item.get("answer", "")))
        prediction = str(item.get("prediction", ""))
        if "Tell me more" in question or "background information" in question:
            continue
        if item.get("substring") not in (0, 0.0, False):
            continue
        if len(gold) > args.max_gold_len:
            continue
        short_wrong += 1
        pid = str(item.get("id", ""))
        entry = qa_by_id.get(pid, {}).get(question)
        if not entry:
            unmapped += 1
            continue
        key, _ = entry
        category, matches = categorize(
            pid, key, prediction, profile_answers, answers_by_key
        )
        counts[category] = counts.get(category, 0) + 1
        if len(examples) < args.topk:
            examples.append(
                {
                    "category": category,
                    "id": pid,
                    "key": key,
                    "ground_truth": gold,
                    "prediction": prediction,
                    "matches": matches,
                }
            )
    result = {
        "details": args.details,
        "short_wrong": short_wrong,
        "unmapped": unmapped,
        "counts": counts,
        "examples": examples,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
