#!/usr/bin/env python3
"""C4-b: 逐题跨模态表示余弦 vs 该题的 (IT✓, PT✓)。

对评估用的同题 IT/PT (Fill/cloze), 用 base 模型取逐层 mean-pool 表示,
逐题算 cos(rep_IT, rep_PT), 再按 p02_eval_origin 的逐题明细把题分成
(TT)=两模态都对 / (TF,FT)=单侧 / (FF)=都错, 比较各组的跨模态共享分量。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..eval.eval_vllm import fill_jobs
from .patching import batch_to_device, build_batch


@torch.no_grad()
def rep_batch(model, items, modality, layers, device, processor, max_length, pool):
    batch = build_batch(items, modality, processor, max_length)
    ids, attn, pixel, labels = batch_to_device(batch, modality, device)
    o = model(input_ids=ids, attention_mask=attn, pixel_values=pixel,
              output_hidden_states=True, use_cache=False)
    mask = attn.unsqueeze(-1).float()
    amask = (labels != -100).unsqueeze(-1).float()
    wmask = amask if (pool == "answer" and amask.sum() > 0) else mask
    pos = (mask.sum(1).long() - 1).clamp_min(0)
    idx = torch.arange(mask.shape[0], device=mask.device)
    out = {}
    for L in layers:
        h = o.hidden_states[L].float()
        out[L] = (h[idx, pos].cpu() if pool == "last"
                  else ((h * wmask).sum(1) / wmask.sum(1).clamp_min(1)).cpu())
    return out


def run_jobs(model, jobs, modality, layers, device, processor, max_length, batch_size, pool):
    out = {L: [] for L in layers}
    for b in range(0, len(jobs), batch_size):
        chunk = jobs[b:b + batch_size]
        items = [{"image": j.image, "question": j.question,
                  "answer": j.metadata["ground_truth"]} for j in chunk]
        r = rep_batch(model, items, modality, layers, device, processor, max_length, pool)
        for L in layers:
            out[L].append(r[L])
    return {L: torch.cat(v, 0) for L, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data_split_folder", required=True)
    ap.add_argument("--celebrity_data", required=True)
    ap.add_argument("--forget_ratio", type=int, default=5)
    ap.add_argument("--scope", choices=["forget", "retain_shared", "retain_celebrity"],
                    default="forget")
    ap.add_argument("--details", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,28,31,32")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--pool", choices=["mean", "last", "answer"], default="mean")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    forget_file = str(Path(args.data_split_folder) / f"forget_{args.forget_ratio}"
                      / "train-00000-of-00001.parquet")
    retain_file = str(Path(args.data_split_folder) / f"retain_{100 - args.forget_ratio}"
                      / "train-00000-of-00001.parquet")
    path, mode = {
        "forget": (forget_file, "forget"),
        "retain_shared": (retain_file, "retain_shared"),
        "retain_celebrity": (args.celebrity_data, "retain_celebrity"),
    }[args.scope]

    eargs = SimpleNamespace(max_questions_per_group=None, prompt_suffix="",
                            image_noise_sigma=0.0, max_eval_samples=None)
    jobs = fill_jobs(path, mode, forget_file, eargs)
    it_jobs = [j for j in jobs if j.metadata["type"] == "Image_Textual"]
    pt_jobs = [j for j in jobs if j.metadata["type"] == "Pure_Text"]
    it_pos = {(j.metadata["image_id"], j.metadata["qid"]): i for i, j in enumerate(it_jobs)}
    pt_pos = {(j.metadata["image_id"], j.metadata["qid"]): i for i, j in enumerate(pt_jobs)}

    det = json.load(open(args.details))["Fill_Questions"]
    ok = {}
    for d in det:
        ok.setdefault((d["image_id"], d["qid"]), {})[d["question type"]] = bool(d["correct"])

    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device).eval()
    model.config.use_cache = False

    it_reps = run_jobs(model, it_jobs, "mm", layers, args.device, processor,
                       args.max_length, args.batch_size, args.pool)
    pt_reps = run_jobs(model, pt_jobs, "um", layers, args.device, processor,
                       args.max_length, args.batch_size, args.pool)

    groups = {"TT": [], "one": [], "FF": []}
    for key in sorted(set(it_pos) & set(pt_pos) & set(ok)):
        flag = ok[key]
        it_ok, pt_ok = flag.get("Image_Textual"), flag.get("Pure_Text")
        if it_ok is None or pt_ok is None:
            continue
        label = "TT" if (it_ok and pt_ok) else ("FF" if not it_ok and not pt_ok else "one")
        groups[label].append(key)

    per_group = {}
    for label, keys in groups.items():
        entry = {}
        for L in layers:
            cs = []
            for key in keys:
                a = it_reps[L][it_pos[key]]
                b = pt_reps[L][pt_pos[key]]
                cs.append(torch.nn.functional.cosine_similarity(a[None], b[None]).item())
            entry[str(L)] = {"cos": (sum(cs) / len(cs) if cs else None), "n": len(cs)}
        per_group[label] = entry

    result = {"scope": args.scope, "pool": args.pool, "layers": layers,
              "counts": {k: len(v) for k, v in groups.items()},
              "groups": per_group}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"scope={args.scope} pool={args.pool} counts={result['counts']}")
    for L in layers:
        row = " ".join(f"{lab}:{per_group[lab][str(L)]['cos']:.3f}"
                       if per_group[lab][str(L)]["cos"] is not None else f"{lab}:--"
                       for lab in ("TT", "one", "FF"))
        print(f"{L:>5} | {row}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
