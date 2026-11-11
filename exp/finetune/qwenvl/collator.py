"""Collator for Qwen3-VL supervised finetuning (standard padding).

Adapted from QwenLM/Qwen3-VL qwen-vl-finetune/qwenvl/data/data_processor.py
(Apache-2.0). Video packing/flattening paths are intentionally omitted.
"""

import torch


def _pad_position_ids(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded.append(torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1))
    return torch.cat(padded, dim=1)


class DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer, include_position_ids: bool = True):
        self.tokenizer = tokenizer
        self.include_position_ids = include_position_ids

    def __call__(self, instances):
        input_ids = [instance["input_ids"].squeeze(0) for instance in instances]
        labels = [instance["labels"].squeeze(0) for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        max_length = self.tokenizer.model_max_length
        input_ids = input_ids[:, :max_length]
        labels = labels[:, :max_length]

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if self.include_position_ids:
            position_ids = _pad_position_ids([instance["position_ids"] for instance in instances])
            batch["position_ids"] = position_ids[:, :, :max_length]

        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat(
                [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance],
                dim=0,
            )
        return batch
