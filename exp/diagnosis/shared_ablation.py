#!/usr/bin/env python3
"""C3: 投影掉"跨模态共享子空间"后, 通用任务(retain/real) vs 遗忘任务的 CE 损伤。

共享子空间: 对 forget_5 的 n 个配对, 取层 L 的 IT/PT 表示, 用两模态子空间之间的
主角度方向(共享方向 U_it, U_pt)近似。干预: 在层 L 前向时把各自模态的共享方向投影掉
(h <- h - h U U^T)。度量: 各 scope 的答案 token CE(teacher-forced), 看 retain/real 的
CE 上升是否显著大于 forget。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..eval.eval_vllm import fill_jobs
from ..unlearn._paired import build_pairs
from .patching import batch_to_device, build_batch, get_layers


def probe_pairs(data_split_dir, split, n, seed=42):
    ps = build_pairs(pd.read_parquet(Path(data_split_dir) / split / "train-00000-of-00001.parquet"))
    order = torch.randperm(len(ps), generator=torch.Generator().manual_seed(seed)).tolist()
    return [ps[i] for i in order[:n]]


@torch.no_grad()
def reps(model, pairs, modality, layer, device, processor, max_length, batch_size):
    chunks = []
    for b in range(0, len(pairs), batch_size):
        chunk = pairs[b:b + batch_size]
        items = [{"image": p["image"] if modality == "mm" else None,
                  "question": p["mm_q"] if modality == "mm" else p["um_q"],
                  "answer": p["mm_a"] if modality == "mm" else p["um_a"]} for p in chunk]
        batch = build_batch(items, modality, processor, max_length)
        ids, attn, pixel, _ = batch_to_device(batch, modality, device)
        o = model(input_ids=ids, attention_mask=attn, pixel_values=pixel,
                  output_hidden_states=True, use_cache=False)
        mask = attn.unsqueeze(-1).float()
        h = o.hidden_states[layer].float()
        chunks.append(((h * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu())
    return torch.cat(chunks, 0)


def shared_dirs(A, B, k):
    Ac = A.float() - A.float().mean(0, keepdim=True)
    Bc = B.float() - B.float().mean(0, keepdim=True)
    Qa, _ = torch.linalg.qr(Ac.T)
    Qb, _ = torch.linalg.qr(Bc.T)
    U, S, V = torch.linalg.svd(Qa.T @ Qb)
    return (Qa @ U[:, :k]), (Qb @ V[:, :k]), S[:k]


def ce_over(model, jobs, modality, device, processor, max_length, batch_size):
    tot = 0.0
    ntok = 0
    for b in range(0, len(jobs), batch_size):
        chunk = jobs[b:b + batch_size]
        items = [{"image": j.image, "question": j.question,
                  "answer": j.metadata["ground_truth"]} for j in chunk]
        batch = build_batch(items, modality, processor, max_length)
        ids, attn, pixel, labels = batch_to_device(batch, modality, device)
        logits = model(input_ids=ids, attention_mask=attn, pixel_values=pixel).logits
        lg = logits[:, :-1].float()
        sl = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.shape[-1]), sl.reshape(-1), ignore_index=-100, reduction="sum")
        tot += float(loss)
        ntok += int((sl != -100).sum())
    return tot / max(ntok, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data_split_folder", required=True)
    ap.add_argument("--celebrity_data", required=True)
    ap.add_argument("--forget_ratio", type=int, default=5)
    ap.add_argument("--output", required=True)
    ap.add_argument("--layers", default="16,31")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--scopes", default="forget,retain_shared,retain_celebrity")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    forget_file = str(Path(args.data_split_folder) / f"forget_{args.forget_ratio}"
                      / "train-00000-of-00001.parquet")
    retain_file = str(Path(args.data_split_folder) / f"retain_{100 - args.forget_ratio}"
                      / "train-00000-of-00001.parquet")
    scope_paths = {"forget": (forget_file, "forget"),
                   "retain_shared": (retain_file, "retain_shared"),
                   "retain_celebrity": (args.celebrity_data, "retain_celebrity")}
    eargs = SimpleNamespace(max_questions_per_group=None, prompt_suffix="",
                            image_noise_sigma=0.0, max_eval_samples=None)

    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device).eval()
    model.config.use_cache = False
    blocks = get_layers(model)

    pairs = probe_pairs(args.data_split_folder, f"forget_{args.forget_ratio}", args.n, args.seed)
    scopes = [s.strip() for s in args.scopes.split(",") if s.strip()]
    jobs = {}
    for sc in scopes:
        path, mode = scope_paths[sc]
        js = fill_jobs(path, mode, forget_file, eargs)
        jobs[sc] = {"Image_Textual": [j for j in js if j.metadata["type"] == "Image_Textual"],
                    "Pure_Text": [j for j in js if j.metadata["type"] == "Pure_Text"]}

    result = {"layers": {}, "k": args.k, "n": args.n}
    for L in [int(x) for x in args.layers.split(",")]:
        A = reps(model, pairs, "mm", L, args.device, processor, args.max_length, args.batch_size)
        B = reps(model, pairs, "um", L, args.device, processor, args.max_length, args.batch_size)
        u_it, u_pt, sv = shared_dirs(A, B, args.k)
        d = u_it.shape[1]
        sv = sv.tolist()

        base_ce, cut_ce = {}, {}
        for kind, dirs in (("Image_Textual", u_it), ("Pure_Text", u_pt)):
            for sc in scopes:
                base_ce[(sc, kind)] = ce_over(model, jobs[sc][kind], "mm" if kind == "Image_Textual" else "um",
                                              args.device, processor, args.max_length, args.batch_size)
            U = dirs.to(args.device, torch.float32)

            def hook(module, inp, U=U):
                hs = inp[0]
                f = hs.float()
                proj = (f @ U) @ U.T
                return ((f - proj).to(hs.dtype),) + inp[1:]

            h = blocks[L].register_forward_pre_hook(hook)
            for sc in scopes:
                cut_ce[(sc, kind)] = ce_over(model, jobs[sc][kind], "mm" if kind == "Image_Textual" else "um",
                                             args.device, processor, args.max_length, args.batch_size)
            h.remove()

        entry = {"k": d, "svals": sv, "scopes": {}}
        for sc in scopes:
            e = {}
            for kind in ("Image_Textual", "Pure_Text"):
                b0 = base_ce[(sc, kind)]
                b1 = cut_ce[(sc, kind)]
                e[kind] = {"ce_base": b0, "ce_cut": b1, "delta": b1 - b0}
            entry["scopes"][sc] = e
        result["layers"][str(L)] = entry
        print(f"layer {L}: " + " | ".join(
            f"{sc} dCE_IT={entry['scopes'][sc]['Image_Textual']['delta']:+.3f} "
            f"dCE_PT={entry['scopes'][sc]['Pure_Text']['delta']:+.3f}" for sc in scopes), flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
