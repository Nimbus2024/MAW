#!/usr/bin/env python3
"""同题配对数据层：把一道题的 VQA(multimodal) 与 QA(unimodal) 绑成一个样本，
由同一个 DataLoader 组织 batch，collate 时产出位置一一对应的 mm/um batch。

依赖前提（构建时断言）：每行 MM_QA 与 UM_QA 的 question 字典 key 集合、顺序、数量一致。
每个训练 step 内的 mm[i] 与 um[i] 恒为同一道题。

数据核对(UMU-bench 训练分片, MM_QA vs UM_QA 同 (row,key))：12 个 key 中除 `Name` 外
答案完全一致、提问仅差人称("this person"/人名)；`Name` 例外: MM 短问短答(人名) vs
UM 长问长 bio——仍视为同一人物身份知识、保留成对。评测列(Classify/Cloze/Generation)
为测试专用, 不参与训练；biography 官方训练亦不使用。

内容形态：
  plain: {mm:{image,question,answer}, um:{question,answer}}   (GA/KLmin/simNPO/retrain)
  dpo  : 额外每题共用同一 idk: mm/um 各带 answer_plus(forget) 与 answer_0(idk)
"""
import ast
import random

import pandas as pd
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset

from .unlearn_dataset import mask_prompt_labels


def _load(value):
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def _json2token(obj, sort_json_key=True):
    if isinstance(obj, dict):
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        output = ""
        keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
        for k in keys:
            output += f"<s_{k}>" + _json2token(obj[k], sort_json_key) + f"</s_{k}>"
        return output
    if isinstance(obj, list):
        return "<sep/>".join([_json2token(item, sort_json_key) for item in obj])
    return str(obj)


def build_pairs(df: pd.DataFrame, sort_json_key=True):
    """逐行校验 MM_QA/UM_QA key 对称并展开为 (row, key) 的成对样本。

    每项: {image, mm_q, mm_a, um_q, um_a}(均为 token 化文本; image 为 PIL)。
    """
    pairs = []
    problems = []
    for idx, row in df.iterrows():
        try:
            image_data = row["image"].get("bytes")
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception:
            problems.append((idx, "image load failed"))
            continue
        try:
            mm = _load(row["MM_QA"])
            um = _load(row["UM_QA"])
        except Exception as exc:
            problems.append((idx, f"parse failed: {exc}"))
            continue
        mm_k = list(mm.get("question", {}).keys())
        um_k = list(um.get("question", {}).keys())
        if mm_k != um_k:
            problems.append(
                (idx, f"MM_QA/UM_QA key mismatch: mm={len(mm_k)} um={len(um_k)} "
                      f"order_equal={mm_k == um_k} set_equal={set(mm_k) == set(um_k)}"))
            continue
        mm_qs = mm["question"]
        mm_as = mm["answer"]
        um_qs = um["question"]
        um_as = um["answer"]
        for k in mm_k:
            pairs.append({
                "row": idx,
                "key": k,
                "image": image,
                "mm_q": _json2token(mm_qs[k], sort_json_key),
                "mm_a": _json2token(mm_as[k], sort_json_key),
                "um_q": _json2token(um_qs[k], sort_json_key),
                "um_a": _json2token(um_as[k], sort_json_key),
            })
    if problems:
        raise ValueError(
            "PairedSource: 存在无法严格同题配对的样本(前几条): " +
            "; ".join(f"row{idx}: {msg}" for idx, msg in problems[:10]) +
            f" (共 {len(problems)} 条)。请先核对数据 MM_QA/UM_QA 对称性。")
    if not pairs:
        raise ValueError("PairedSource: 无任何配对样本。")
    return pairs


class PairedDataset(Dataset):
    def __init__(self, pairs, dpo=False, idk_list=None):
        if dpo and not idk_list:
            raise ValueError("dpo=True 需要 idk_list")
        self.pairs = pairs
        self.dpo = dpo
        self.idk_list = idk_list

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        if not self.dpo:
            return {
                "mm": {"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]},
                "um": {"question": p["um_q"], "answer": p["um_a"]},
            }
        idk = random.choice(self.idk_list)
        return {
            "mm": {"image": p["image"], "question": p["mm_q"],
                   "answer_plus": p["mm_a"], "answer_0": idk},
            "um": {"question": p["um_q"], "answer_plus": p["um_a"], "answer_0": idk},
        }


def collate_plain(batch, processor, args):
    from .unlearn_dataset import train_collate_fn_llava_multimodal, train_collate_fn_llava_unimodal
    mm = [b["mm"] for b in batch]
    um = [b["um"] for b in batch]
    return {
        "mm": train_collate_fn_llava_multimodal(mm, processor, args),
        "um": train_collate_fn_llava_unimodal(um, processor, args),
    }


def collate_dpo(batch, processor, args):
    mm_w, mm_l, mm_img, mm_aw, mm_al = [], [], [], [], []
    um_w, um_l, um_aw, um_al = [], [], [], []
    for b in batch:
        m, u = b["mm"], b["um"]
        mm_img.append(m["image"])
        q = m["question"]
        mm_w.append(f"USER: <image>\n{q}\nASSISTANT: {m['answer_0']}")
        mm_l.append(f"USER: <image>\n{q}\nASSISTANT: {m['answer_plus']}")
        mm_aw.append(m["answer_0"])
        mm_al.append(m["answer_plus"])
        uq = u["question"]
        um_w.append(f"USER: {uq}\nASSISTANT: {u['answer_0']}")
        um_l.append(f"USER: {uq}\nASSISTANT: {u['answer_plus']}")
        um_aw.append(u["answer_0"])
        um_al.append(u["answer_plus"])
    mm_bw, mm_bl = _pair_prompt(mm_w, mm_l, mm_img, processor, args, mm_aw, mm_al)
    um_bw, um_bl = _pair_prompt(um_w, um_l, None, processor, args, um_aw, um_al)
    return {"mm": {"batch_w": mm_bw, "batch_l": mm_bl},
            "um": {"batch_w": um_bw, "batch_l": um_bl}}


def _pair_prompt(texts_w, texts_l, images, processor, args, answers_w, answers_l):
    kw = dict(padding=True, truncation=True, max_length=args.max_length,
              add_special_tokens=False, return_tensors="pt")
    if images:
        kw["images"] = images
        kw["size"] = {"shortest_edge": 336}
    bw = processor(text=texts_w, **kw)
    bl = processor(text=texts_l, **kw)
    return ({
        "input_ids": bw["input_ids"], "attention_mask": bw["attention_mask"],
        "pixel_values": bw["pixel_values"] if images else None,
        "labels": mask_prompt_labels(bw, processor, answers_w),
    }, {
        "input_ids": bl["input_ids"], "attention_mask": bl["attention_mask"],
        "pixel_values": bl["pixel_values"] if images else None,
        "labels": mask_prompt_labels(bl, processor, answers_l),
    })
