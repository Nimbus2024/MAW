"""Parquet-native Qwen3-VL datasets for UMU-bench.

Uses the official Qwen3-VL chat template and assistant-only masking from
qwenvl.processor, with 3D mRoPE position ids from qwenvl.rope2d.
"""

import ast
import math
import random
import re
from io import BytesIO
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

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


def iter_qa_with_keys(value):
    data = _as_mapping(value)
    questions = data.get("question", {}) or {}
    answers = data.get("answer", {}) or {}
    if not isinstance(questions, dict):
        return
    for key in questions:
        yield str(key), json2token(questions[key]), json2token(answers.get(key, ""))


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


def _norm_value(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


class UmuHardNegativeDataset(Dataset):
    def __init__(
        self,
        parquet_path,
        processor,
        modalities: str = "mm_um",
        max_samples=None,
        hard_neg_k: int = 4,
        exclude_keys=("Name", "Description"),
        seed: int = 42,
    ):
        self.processor = processor
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.hard_neg_k = hard_neg_k
        self.seed = seed
        df = pd.read_parquet(parquet_path).reset_index(drop=True)

        exclude_keys = {str(key) for key in exclude_keys}
        values_by_key = {}
        for row_idx, row in df.iterrows():
            pid = str(row.get("ID"))
            answers = {}
            for column in ("UM_QA", "MM_QA"):
                for key, _, answer in iter_qa_with_keys(row.get(column)):
                    answers.setdefault(key, answer)
            for key, answer in answers.items():
                if key in exclude_keys or not str(answer).strip():
                    continue
                values_by_key.setdefault(key, []).append((pid, answer))

        items = []
        for row_idx, row in df.iterrows():
            for column, kind in (("MM_QA", "Image_Textual"), ("UM_QA", "Pure_Text")):
                if kind == "Image_Textual" and "mm" not in modalities:
                    continue
                if kind == "Pure_Text" and "um" not in modalities:
                    continue
                for key, question, answer in iter_qa_with_keys(row.get(column)):
                    if key in exclude_keys:
                        continue
                    negatives = self._sample_negatives(
                        str(row.get("ID")),
                        key,
                        answer,
                        values_by_key,
                        row_idx,
                    )
                    if not negatives:
                        continue
                    items.append(
                        {
                            "row_idx": row_idx,
                            "id": str(row.get("ID")),
                            "key": key,
                            "question": question,
                            "answer": answer,
                            "negatives": negatives,
                            "type": kind,
                        }
                    )
        if max_samples is not None:
            random.Random(seed).shuffle(items)
            items = items[:max_samples]

        images = {idx: _decode_image(df.iloc[idx]) for idx in {item["row_idx"] for item in items}}
        self.samples = [
            {
                "id": item["id"],
                "key": item["key"],
                "image": images[item["row_idx"]] if item["type"] == "Image_Textual" else None,
                "question": item["question"],
                "answer": item["answer"],
                "negatives": item["negatives"],
                "type": item["type"],
            }
            for item in items
        ]
        print(
            f"UmuHardNegativeDataset: {len(self.samples)} samples, "
            f"k={self.hard_neg_k}, modalities={modalities}"
        )

    def _sample_negatives(self, pid, key, answer, values_by_key, row_idx):
        correct = _norm_value(answer)
        pool = []
        seen = set()
        for other_id, value in values_by_key.get(key, []):
            if other_id == pid:
                continue
            normalized = _norm_value(value)
            if not normalized or normalized == correct or normalized in seen:
                continue
            seen.add(normalized)
            pool.append(value)
        if not pool:
            return []
        rng = random.Random(self.seed + row_idx + sum(ord(ch) for ch in key))
        if len(pool) <= self.hard_neg_k:
            return pool
        return rng.sample(pool, self.hard_neg_k)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        candidates = []
        for answer in [sample["answer"]] + sample["negatives"]:
            messages = build_messages(sample["question"], answer, sample["image"])
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
            candidates.append(item)
        return {"candidates": candidates, "num_candidates": len(candidates)}


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


def load_task_samples(parquet_path, limit=None, seed: int = 42):
    df = pd.read_parquet(parquet_path).reset_index(drop=True)
    items = []
    images = {}
    for row_idx, row in df.iterrows():
        for task, column in (
            ("classification", "Classify"),
            ("cloze", "Cloze"),
            ("generation", "Generation"),
        ):
            data = _as_mapping(row.get(column))
            for group_name, group in data.items():
                if not isinstance(group, dict):
                    continue
                modality = "Image_Textual" if group_name == "muitimodal" else "Pure_Text"
                for item in group.values():
                    if not isinstance(item, dict):
                        continue
                    question = item.get("question", "")
                    answer = item.get("answer", "")
                    if not question or answer in (None, ""):
                        continue
                    options = item.get("options") if task == "classification" else None
                    correct_key = None
                    if task == "classification":
                        if not isinstance(options, dict) or not options:
                            continue
                        correct_key = str(answer).split(".", 1)[0].strip().upper()
                        if correct_key not in {str(key).upper() for key in options}:
                            continue
                        answer_text = str(options.get(correct_key, answer))
                        prompt = (
                            f"{question}\n"
                            + "\n".join(f"{key}. {value}" for key, value in options.items())
                        )
                    elif task == "cloze":
                        answer_text = str(answer)
                        prompt = (
                            question.replace("__", "[Blank]")
                            + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
                        )
                    else:
                        answer_text = str(answer)
                        prompt = question
                    if modality == "Image_Textual":
                        if row_idx not in images:
                            images[row_idx] = _decode_image(row)
                        image = images[row_idx]
                    else:
                        image = None
                    items.append(
                        {
                            "id": str(row.get("ID")),
                            "image": image,
                            "task": task,
                            "type": modality,
                            "question": prompt,
                            "answer": answer_text,
                            "options": options,
                            "correct_key": correct_key,
                        }
                    )
    if limit is not None and limit < len(items):
        items = random.Random(seed).sample(items, limit)
    return items



class ModalityBalancedBatchSampler(Sampler):
    def __init__(self, dataset, mm_per_batch: int, um_per_batch: int, seed: int = 42):
        if mm_per_batch < 1 or um_per_batch < 1:
            raise ValueError("mm_per_batch and um_per_batch must be positive")
        self.mm_indices = [
            index for index, sample in enumerate(dataset.samples)
            if sample["type"] == "Image_Textual"
        ]
        self.um_indices = [
            index for index, sample in enumerate(dataset.samples)
            if sample["type"] == "Pure_Text"
        ]
        if not self.mm_indices or not self.um_indices:
            raise ValueError("balanced sampler requires both multimodal and unimodal samples")
        self.mm_per_batch = mm_per_batch
        self.um_per_batch = um_per_batch
        self.seed = seed
        self.epoch = 0
        self.num_batches = max(
            math.ceil(len(self.mm_indices) / mm_per_batch),
            math.ceil(len(self.um_indices) / um_per_batch),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        self.epoch += 1
        rng = random.Random(self.seed + self.epoch)
        mm = list(self.mm_indices)
        um = list(self.um_indices)
        rng.shuffle(mm)
        rng.shuffle(um)
        mm_pos = 0
        um_pos = 0
        for _ in range(self.num_batches):
            batch = []
            for _ in range(self.mm_per_batch):
                if mm_pos >= len(mm):
                    rng.shuffle(mm)
                    mm_pos = 0
                batch.append(mm[mm_pos])
                mm_pos += 1
            for _ in range(self.um_per_batch):
                if um_pos >= len(um):
                    rng.shuffle(um)
                    um_pos = 0
                batch.append(um[um_pos])
                um_pos += 1
            yield batch
