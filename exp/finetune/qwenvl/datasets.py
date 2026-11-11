"""Parquet-native Qwen3-VL datasets for UMU-bench.

Uses the official Qwen3-VL chat template and assistant-only masking from
qwenvl.processor, with 3D mRoPE position ids from qwenvl.rope2d.
"""

import ast
import random
from io import BytesIO
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .processor import build_messages, preprocess_qwen_visual
from .rope2d import get_rope_index_3


def json2token(obj: Any, sort_json_key: bool = True) -> str:
    if isinstance(obj, dict):
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        output = ""
        keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
        for key in keys:
            output += f"<s_{key}>" + json2token(obj[key], sort_json_key) + f"</s_{key}>"
        return output
    if isinstance(obj, list):
        return "<sep/>".join(json2token(item, sort_json_key) for item in obj)
    return str(obj)


def _as_mapping(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return value if isinstance(value, dict) else {}


def iter_qa(value):
    data = _as_mapping(value)
    questions = data.get("question", {}) or {}
    answers = data.get("answer", {}) or {}
    if not isinstance(questions, dict):
        return
    for key in questions:
        yield json2token(questions[key]), json2token(answers.get(key, ""))


def _decode_image(row) -> Image.Image:
    return Image.open(BytesIO(row["image"]["bytes"])).convert("RGB")


def _qa_items(df, modality: str):
    items = []
    for row_idx, row in df.iterrows():
        if modality == "mm":
            for question, answer in iter_qa(row.get("MM_QA")):
                items.append((row_idx, question, answer, "Image_Textual"))
        elif modality == "um":
            for question, answer in iter_qa(row.get("UM_QA")):
                items.append((row_idx, question, answer, "Pure_Text"))
        else:
            raise ValueError(f"unsupported modality: {modality}")
    return items


class UmuQwenDataset(Dataset):
    def __init__(self, parquet_path, processor, modalities: str = "mm", max_samples=None):
        self.processor = processor
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        df = pd.read_parquet(parquet_path).reset_index(drop=True)

        items = []
        if "mm" in modalities:
            items.extend(_qa_items(df, "mm"))
        if "um" in modalities:
            items.extend(_qa_items(df, "um"))
        if max_samples is not None:
            items = items[:max_samples]

        images = {idx: _decode_image(df.iloc[idx]) for idx, _, _, _ in items}
        self.samples = [
            {"image": images[row_idx], "question": question, "answer": answer, "type": kind}
            for row_idx, question, answer, kind in items
        ]
        print(f"UmuQwenDataset: {len(self.samples)} QA samples from {parquet_path} ({modalities})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = build_messages(sample["question"], sample["answer"], sample["image"])
        data = preprocess_qwen_visual(messages, self.processor)
        input_ids = data["input_ids"]
        position_ids, _ = get_rope_index_3(
            self.merge_size,
            input_ids,
            image_grid_thw=data.get("image_grid_thw"),
            attention_mask=torch.ones_like(input_ids),
        )
        item = {
            "input_ids": input_ids,
            "labels": data["labels"],
            "position_ids": position_ids,
        }
        if "pixel_values" in data:
            item["pixel_values"] = data["pixel_values"]
        if "image_grid_thw" in data:
            item["image_grid_thw"] = data["image_grid_thw"]
        return item


def load_eval_samples(parquet_path, modality: str = "mm", limit=None, seed: int = 42):
    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    items = _qa_items(df, modality)
    if limit is not None and limit < len(items):
        items = random.Random(seed).sample(items, limit)
    row_ids = {row_idx for row_idx, _, _, _ in items}
    images = {row_idx: _decode_image(df.iloc[row_idx]) for row_idx in row_ids}
    return [
        {
            "id": str(df.iloc[row_idx]["ID"]),
            "image": images[row_idx],
            "question": question,
            "answer": answer,
            "type": kind,
        }
        for row_idx, question, answer, kind in items
    ]
