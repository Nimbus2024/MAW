#!/usr/bin/env python3
"""真·多选题 CE：在候选选项之间做 4 路 softmax。

与 ce_breakdown.py 的 classification 口径（生成正确选项**文本**的 token CE）不同,
这里对每个选项算**长度归一化的序列 log-prob**, 再在选项间做 softmax:

    p(c) = softmax_c lm_norm(option_c)
    CE_MC = -log p(correct)

这样 "约束型多选" 才名副其实（只在候选集内归一化, ∈[0, ln K]）。
同时报告 argmax 命中率（= 模型的选项准确率）。

数据: UMU-bench forget_5 的 `Classify` 列（muitimodal / unimodal 两组）。
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, LlavaForConditionalGeneration


def _literal(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def build_prompt(question, options, include_image):
    opts = " ".join(str(v) for v in options.values())
    q = f"{question}\nSelect answer in [{opts}]"
    prefix = "USER: <image>\n" if include_image else "USER: "
    return prefix + f"{q}\nASSISTANT: "


@torch.no_grad()
def score_option(model, processor, prompt, cont, image, device):
    """返回 (logp_sum, n_tokens): 给定 prompt 续写 cont 的长度归一化前 log-prob。"""
    full = prompt + cont
    kw = dict(return_tensors="pt", add_special_tokens=False)
    enc_full = processor(text=[full], images=[image] if image is not None else None, **kw)
    enc_prompt = processor(text=[prompt], images=[image] if image is not None else None, **kw)
    full_ids = enc_full["input_ids"][0]
    prompt_ids = enc_prompt["input_ids"][0]
    n = min(len(full_ids), len(prompt_ids))
    i = 0
    while i < n and full_ids[i].item() == prompt_ids[i].item():
        i += 1
    n_prompt = i
    if n_prompt == 0 or n_prompt >= len(full_ids):
        return float("nan"), 0
    ids = enc_full["input_ids"].to(device)
    pixel = enc_full.get("pixel_values")
    pixel = pixel.to(device) if pixel is not None else None
    attn = enc_full["attention_mask"].to(device)
    logits = model(input_ids=ids, attention_mask=attn, pixel_values=pixel).logits
    lp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    tgt = ids[0, 1:]
    lps = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    sl = lps[n_prompt - 1:]
    return float(sl.sum()), int(sl.numel())


@torch.no_grad()
def run_group(model, processor, rows, include_image, device):
    ces, hits = [], []
    for r in rows:
        options = r["options"]
        keys = list(options.keys())
        if not keys:
            continue
        prompt = build_prompt(r["question"], options, include_image)
        scores = []
        for k in keys:
            s, nt = score_option(model, processor, prompt, str(options[k]), r["image"], device)
            scores.append(s / nt if nt > 0 else float("-inf"))
        st = torch.tensor(scores, dtype=torch.float64)
        p = torch.softmax(st, dim=0)
        idx = keys.index(r["correct_key"])
        pc = float(p[idx].clamp_min(1e-12))
        ces.append(-torch.log(torch.tensor(pc)).item())
        hits.append(int(int(torch.argmax(st)) == idx))
    return ces, hits


def main():
    ap = argparse.ArgumentParser(description="真 MC CE (4 路 softmax)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--data_split_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device).eval()
    model.config.use_cache = False

    df = pd.read_parquet(Path(args.data_split_dir) / "forget_5" / "train-00000-of-00001.parquet")
    groups = {"it": [], "pt": []}
    for _, row in df.iterrows():
        obj = _literal(row.get("Classify", "{}")) or {}
        img = Image.open(BytesIO(row["image"].get("bytes"))).convert("RGB")
        for side, key in (("muitimodal", "it"), ("unimodal", "pt")):
            for v in (obj.get(side, {}) or {}).values():
                opts = v.get("options", {}) or {}
                ans = str(v.get("answer", ""))
                ck = ans.split(".", 1)[0].strip().upper()
                if ck not in {str(k).upper() for k in opts}:
                    ck = next((k for k in opts if str(opts[k]) == ans), None)
                if ck is None:
                    continue
                groups[key].append({"question": v.get("question", ""), "options": opts,
                                    "correct_key": ck, "image": img if key == "it" else None})

    res = {}
    for key in ("it", "pt"):
        ces, hits = run_group(model, processor, groups[key], key == "it", args.device)
        n = len(ces)
        res[key] = {"ce_mc": sum(ces) / n if n else float("nan"),
                    "acc": sum(hits) / n if n else float("nan"), "n": n}
        print(f"{key}: CE_MC={res[key]['ce_mc']:.4f} acc={res[key]['acc']:.3f} n={n}")
    res["ratio_ce_mc"] = res["pt"]["ce_mc"] / res["it"]["ce_mc"] if res["it"]["ce_mc"] > 0 else float("inf")
    print(f"ratio CE_MC (PT/IT) = {res['ratio_ce_mc']:.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
