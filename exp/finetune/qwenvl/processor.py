"""Qwen3-VL message construction and assistant-only label masking.

Core logic follows QwenLM/Qwen3-VL qwen-vl-finetune/qwenvl/data/data_processor.py
(Apache-2.0), adapted to in-memory PIL images and this repo's parquet datasets.
"""

import torch

IGNORE_INDEX = -100
DEFAULT_IM_END_TOKEN_ID = 151645
ASSISTANT_HEADER = "<|im_start|>assistant\n"


def update_processor_pixels(processor, min_pixels: int, max_pixels: int):
    ip = processor.image_processor
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = min_pixels
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = max_pixels
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        ip.size["shortest_edge"] = min_pixels
        ip.size["longest_edge"] = max_pixels
    return processor


def build_messages(question: str, answer: str, image=None):
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def build_eval_messages(question: str, image=None):
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def _im_end_token_id(tokenizer) -> int:
    token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if token_id is None or token_id == tokenizer.unk_token_id:
        return DEFAULT_IM_END_TOKEN_ID
    return token_id


def _find_subsequence(sequence, subsequence):
    if not subsequence:
        return []
    size = len(subsequence)
    return [
        index
        for index in range(len(sequence) - size + 1)
        if sequence[index : index + size] == subsequence
    ]


def preprocess_qwen_visual(messages, processor):
    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )
    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)
    input_ids_flat = input_ids[0].tolist()
    length = len(input_ids_flat)
    im_end_id = _im_end_token_id(processor.tokenizer)
    header_ids = processor.tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)

    starts = [position + len(header_ids) for position in _find_subsequence(input_ids_flat, header_ids)]
    if not starts:
        assistant_ids = processor.tokenizer.encode("assistant", add_special_tokens=False)
        if len(assistant_ids) != 1:
            raise ValueError(f"cannot locate assistant header tokens: {assistant_ids}")
        starts = [position + 2 for position, token in enumerate(input_ids_flat) if token == assistant_ids[0]]

    for ans_start in starts:
        ans_end = ans_start
        while ans_end < length and input_ids_flat[ans_end] != im_end_id:
            ans_end += 1
        if ans_end < length:
            labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]

    if not torch.any(labels.ne(IGNORE_INDEX)):
        raise ValueError("assistant answer span not found in chat template output")

    result = {
        key: value
        for key, value in full_result.items()
        if key not in ("token_type_ids", "mm_token_type_ids")
    }
    result["labels"] = labels
    result["input_ids"] = input_ids
    return result

