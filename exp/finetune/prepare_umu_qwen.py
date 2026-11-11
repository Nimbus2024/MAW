"""Preflight and manifest for the Qwen3-VL UMU-bench origin datasets."""

import argparse
import json
import os
from collections import Counter

import pandas as pd

from .qwenvl.datasets import _decode_image, iter_qa


def describe(path, name):
    df = pd.read_parquet(path).reset_index(drop=True)
    mm_counts = [len(list(iter_qa(row.get("MM_QA")))) for _, row in df.iterrows()]
    um_counts = [len(list(iter_qa(row.get("UM_QA")))) for _, row in df.iterrows()]
    image_ok = 0
    image_errors = []
    for row_idx, row in df.iterrows():
        try:
            _decode_image(row)
            image_ok += 1
        except Exception as exc:
            image_errors.append({"row": row_idx, "id": str(row.get("ID")), "error": repr(exc)})
    preview = []
    for _, row in df.head(2).iterrows():
        for question, answer in iter_qa(row.get("MM_QA")):
            preview.append(
                {
                    "id": str(row.get("ID")),
                    "question": question,
                    "answer": answer,
                }
            )
            break
    return {
        "name": name,
        "path": path,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "ids": [str(value) for value in df["ID"].tolist()],
        "mm_qa_total": int(sum(mm_counts)),
        "um_qa_total": int(sum(um_counts)),
        "mm_qa_per_row": dict(sorted(Counter(mm_counts).items())),
        "um_qa_per_row": dict(sorted(Counter(um_counts).items())),
        "image_decode_ok": image_ok,
        "image_decode_errors": image_errors[:5],
        "preview": preview,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Preflight UMU-bench parquet files for Qwen3-VL SFT")
    parser.add_argument("--full_parquet", required=True)
    parser.add_argument("--real_parquet", required=True)
    parser.add_argument("--out_dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    full = describe(args.full_parquet, "full_data")
    real = describe(args.real_parquet, "real_person")
    overlap = sorted(set(full["ids"]) & set(real["ids"]))

    report = {
        "full_data": {key: value for key, value in full.items() if key != "ids"},
        "real_person": {key: value for key, value in real.items() if key != "ids"},
        "id_overlap": overlap,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "manifest.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
