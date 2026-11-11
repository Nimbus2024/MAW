"""Quick fit/generalization check for a trained Qwen3-VL UMU origin.

Samples MM_QA from the training parquet and from the independent real_person
split, generates answers, and reports exact / substring / ROUGE-L scores.
"""

import argparse
import json
import os
import re

import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .qwenvl.datasets import load_eval_samples
from .qwenvl.processor import build_eval_messages, update_processor_pixels
from .qwenvl.runtime import resolve_attn_implementation


def normalize(text: str) -> str:
    text = re.sub(r"</?s_[^>]+>", " ", text)
    text = text.replace("<sep/>", " ")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def score_answer(prediction: str, truth: str, scorer):
    pred = normalize(prediction)
    gold = normalize(truth)
    exact = float(pred == gold)
    substring = float(bool(gold) and gold in pred)
    rouge_l = scorer.score(truth, prediction)["rougeL"].fmeasure
    return exact, substring, rouge_l


def generate_answer(model, processor, device, sample, max_new_tokens: int):
    messages = build_eval_messages(sample["question"], sample["image"])
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    for key in ("token_type_ids", "mm_token_type_ids"):
        inputs.pop(key, None)
    inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True).strip()


def evaluate_split(model, processor, device, samples, name, max_new_tokens, scorer):
    details = []
    totals = {"exact": 0.0, "substring": 0.0, "rougeL": 0.0}
    for sample in tqdm(samples, desc=name):
        prediction = generate_answer(model, processor, device, sample, max_new_tokens)
        exact, substring, rouge_l = score_answer(prediction, sample["answer"], scorer)
        totals["exact"] += exact
        totals["substring"] += substring
        totals["rougeL"] += rouge_l
        details.append(
            {
                "id": sample["id"],
                "type": sample["type"],
                "question": sample["question"],
                "ground_truth": sample["answer"],
                "prediction": prediction,
                "exact": exact,
                "substring": substring,
                "rougeL": rouge_l,
            }
        )
    count = max(len(samples), 1)
    summary = {key: total / count for key, total in totals.items()}
    summary["n"] = len(samples)
    return summary, details


def parse_args():
    parser = argparse.ArgumentParser(description="Check Qwen3-VL UMU origin fit and generalization")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_parquet", required=True)
    parser.add_argument("--real_parquet", required=True)
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_real", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--min_pixels", type=int, default=28 * 28 * 16)
    parser.add_argument("--max_pixels", type=int, default=28 * 28 * 576)
    parser.add_argument(
        "--attn_implementation",
        choices=("auto", "sdpa", "flash_attention_2", "eager"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=os.path.isdir(args.model))
    update_processor_pixels(processor, args.min_pixels, args.max_pixels)
    attn_implementation = resolve_attn_implementation(args.attn_implementation)
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "local_files_only": os.path.isdir(args.model),
    }
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, attn_implementation=attn_implementation, **load_kwargs
        )
    except Exception as exc:
        if attn_implementation != "flash_attention_2":
            raise
        print(f"!! flash_attention_2 failed ({exc}); falling back to sdpa")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, attn_implementation="sdpa", **load_kwargs
        )
    model.eval()
    device = next(model.parameters()).device

    train_samples = load_eval_samples(
        args.train_parquet, modality="mm", limit=args.n_train, seed=args.seed
    )
    real_limit = None if args.n_real <= 0 else args.n_real
    real_samples = load_eval_samples(
        args.real_parquet, modality="mm", limit=real_limit, seed=args.seed
    )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    train_summary, train_details = evaluate_split(
        model, processor, device, train_samples, "train", args.max_new_tokens, scorer
    )
    real_summary, real_details = evaluate_split(
        model, processor, device, real_samples, "real_person", args.max_new_tokens, scorer
    )

    result = {"train": train_summary, "real_person": real_summary, "args": vars(args)}
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summary": result,
                    "train_details": train_details,
                    "real_person_details": real_details,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
