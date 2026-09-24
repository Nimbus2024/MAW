#!/usr/bin/env python3
"""CE 尺度差异的成因分解。

P1: QA scope 逐 qid 的 IT/PT 答案 token CE 与 token 数 —— 分离"答案组成"与"per-token CE"。
P2: 答案完全对称的任务 (Cloze/Classification/Generation) 的 IT/PT CE —— 无组成混淆,
    且 Cloze/Cls 是短答案、Generation 是长答案, 天然形成"短 vs 长"对照。

CE 口径与 exp/diagnosis/patching.py 的 CE 探针一致: 训练侧 collate 构 prompt,
labels 掩掉 prompt, 只对答案 token 求 CE。
"""
from __future__ import annotations

import argparse
import ast
import json
import types
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..unlearn._paired import build_pairs
from .patching import batch_to_device, build_batch


@torch.no_grad()
def ce_stats(model, items, modality, processor, device, max_length, batch_size):
    """逐样本答案 token 的 (CE 均值, token 数)。"""
    ces, nts = [], []
    for b in range(0, len(items), batch_size):
        chunk = items[b:b + batch_size]
        batch = build_batch(chunk, modality, processor, max_length)
        ids, attn, pixel, labels = batch_to_device(batch, modality, device)
        logits = model(input_ids=ids, attention_mask=attn, pixel_values=pixel).logits
        lg = logits[:, :-1].float()
        sl = labels[:, 1:]
        ls = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.size(-1)), sl.reshape(-1), reduction="none").view(sl.shape)
        m = sl != -100
        ces.append(((ls * m).sum(1) / m.sum(1).clamp_min(1)).cpu())
        nts.append(m.sum(1).cpu())
    return torch.cat(ces), torch.cat(nts)


def agg(ce, nt):
    """token 加权 CE + 样本平均 CE + 总 token 数。"""
    n = float(nt.sum())
    return {"tw": float((ce * nt).sum() / n) if n > 0 else float("nan"),
            "mean": float(ce.mean()),
            "n_tok": int(nt.sum()),
            "n": int(ce.numel())}


def p1_qa(model, split_dir, processor, device, max_length, batch_size):
    pairs = build_pairs(pd.read_parquet(Path(split_dir) / "forget_5" / "train-00000-of-00001.parquet"))
    it_items = [{"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]} for p in pairs]
    pt_items = [{"question": p["um_q"], "answer": p["um_a"]} for p in pairs]
    ce_it, nt_it = ce_stats(model, it_items, "mm", processor, device, max_length, batch_size)
    ce_pt, nt_pt = ce_stats(model, pt_items, "um", processor, device, max_length, batch_size)

    keys = [p["key"] for p in pairs]
    order = list(dict.fromkeys(keys))
    per_key = {}
    for k in order:
        idx = [i for i, kk in enumerate(keys) if kk == k]
        same = all(pairs[i]["mm_a"] == pairs[i]["um_a"] for i in idx)
        ii, jj = torch.tensor(idx), torch.tensor(idx)
        per_key[k] = {"n": len(idx), "same_answer": same,
                      "it": agg(ce_it[ii], nt_it[ii]), "pt": agg(ce_pt[jj], nt_pt[jj]),
                      "it_chars": float(sum(len(pairs[i]["mm_a"]) for i in idx) / len(idx)),
                      "pt_chars": float(sum(len(pairs[i]["um_a"]) for i in idx) / len(idx))}
        e = per_key[k]
        e["ratio_tw"] = e["pt"]["tw"] / e["it"]["tw"] if e["it"]["tw"] > 0 else float("inf")

    overall = {"it": agg(ce_it, nt_it), "pt": agg(ce_pt, nt_pt)}
    overall["ratio_tw"] = overall["pt"]["tw"] / overall["it"]["tw"]
    overall["ratio_mean"] = overall["pt"]["mean"] / overall["it"]["mean"]

    same_keys = [k for k in order if per_key[k]["same_answer"]]
    ii = torch.tensor([i for i, kk in enumerate(keys) if kk in same_keys])
    comp = {"it": agg(ce_it[ii], nt_it[ii]), "pt": agg(ce_pt[ii], nt_pt[ii])}
    comp["ratio_tw"] = comp["pt"]["tw"] / comp["it"]["tw"]
    return {"per_key": per_key, "key_order": order, "overall": overall,
            "same_answer_only": comp, "diff_keys": [k for k in order if not per_key[k]["same_answer"]]}


def _literal(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def p2_tasks(model, split_dir, processor, device, max_length, batch_size):
    df = pd.read_parquet(Path(split_dir) / "forget_5" / "train-00000-of-00001.parquet")
    from PIL import Image
    from io import BytesIO

    def images(row):
        return Image.open(BytesIO(row["image"].get("bytes"))).convert("RGB")

    tasks = {}
    spec = {
        "cloze": ("Cloze", lambda v: (v.get("question", "").replace("__", "[Blank]")
                                      + "\nPlease **ONLY** provide the correct answer that should replace the [Blank].")),
        "classification": ("Classify", None),
        "generation": ("Generation", lambda v: (v.get("question", "")
                                                + "\nAnswer the question based on your trained knowledge in one sentence accurately in ENGLISH.")),
    }
    for task, (col, qmap) in spec.items():
        it, pt = [], []
        for _, row in df.iterrows():
            obj = _literal(row.get(col, "{}")) or {}
            img = images(row)
            for side, bucket, use_img in (("muitimodal", it, True), ("unimodal", pt, False)):
                for v in (obj.get(side, {}) or {}).values():
                    q = qmap(v) if qmap else v.get("question", "")
                    ans = v.get("answer", "")
                    if task == "classification":
                        opts = v.get("options", {}) or {}
                        opts_str = " ".join(str(x) for x in opts.values())
                        ck = str(ans).split(".", 1)[0].strip().upper()
                        ans = str(opts.get(ck, opts.get(ck.lower(), ans)))
                        q = f"{v.get('question','')}\nSelect answer in [{opts_str}]"
                    bucket.append({"image": img if use_img else None, "question": q, "answer": str(ans)})
        ce_i, nt_i = ce_stats(model, it, "mm", processor, device, max_length, batch_size)
        ce_p, nt_p = ce_stats(model, pt, "um", processor, device, max_length, batch_size)
        e = {"it": agg(ce_i, nt_i), "pt": agg(ce_p, nt_p)}
        e["ratio_tw"] = e["pt"]["tw"] / e["it"]["tw"] if e["it"]["tw"] > 0 else float("inf")
        e["ratio_mean"] = e["pt"]["mean"] / e["it"]["mean"] if e["it"]["mean"] > 0 else float("inf")
        tasks[task] = e
    return tasks


def main():
    ap = argparse.ArgumentParser(description="CE 尺度差异成因分解")
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None, help="可选: 挂在 base 上的 LoRA (H1 分布外对照评测)")
    ap.add_argument("--data_split_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tasks", default="p1,p2")
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device).eval()
    model.config.use_cache = False
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
        model.config.use_cache = False

    result = {}
    want = args.tasks.split(",")
    if "p1" in want:
        result["p1_qa"] = p1_qa(model, args.data_split_dir, processor, args.device,
                                args.max_length, args.batch_size)
        print("== P1 QA per-key (token-weighted CE) ==")
        print(f"{'key':>18} {'same':>5} {'it_tw':>8} {'pt_tw':>8} {'ratio':>7} {'it_n':>6} {'pt_n':>6} {'it_ch':>6} {'pt_ch':>6}")
        for k in result["p1_qa"]["key_order"]:
            e = result["p1_qa"]["per_key"][k]
            print(f"{k:>18} {str(e['same_answer']):>5} {e['it']['tw']:>8.4f} {e['pt']['tw']:>8.4f} "
                  f"{e['ratio_tw']:>7.2f} {e['it']['n_tok']:>6} {e['pt']['n_tok']:>6} "
                  f"{e['it_chars']:>6.1f} {e['pt_chars']:>6.1f}")
        o = result["p1_qa"]["overall"]
        c = result["p1_qa"]["same_answer_only"]
        print(f"OVERALL  it_tw={o['it']['tw']:.4f} pt_tw={o['pt']['tw']:.4f} ratio={o['ratio_tw']:.2f}")
        print(f"SAME-ANS it_tw={c['it']['tw']:.4f} pt_tw={c['pt']['tw']:.4f} ratio={c['ratio_tw']:.2f}")
        print(f"diff keys: {result['p1_qa']['diff_keys']}")
    if "p2" in want:
        result["p2_tasks"] = p2_tasks(model, args.data_split_dir, processor, args.device,
                                      args.max_length, args.batch_size)
        print("== P2 answer-symmetric tasks ==")
        print(f"{'task':>16} {'it_tw':>8} {'pt_tw':>8} {'ratio':>7} {'it_n':>6} {'pt_n':>6} {'it_ch':>7} {'pt_ch':>7}")
        for t, e in result["p2_tasks"].items():
            print(f"{t:>16} {e['it']['tw']:>8.4f} {e['pt']['tw']:>8.4f} {e['ratio_tw']:>7.2f} "
                  f"{e['it']['n_tok']:>6} {e['pt']['n_tok']:>6} {e['it']['n']:>7} {e['pt']['n']:>7}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
