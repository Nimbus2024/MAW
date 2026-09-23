#!/usr/bin/env python3
"""CE 尺度差异成因的后续验证 (P3/P4/P5)。

P3: 2x2 因子 (参照方式 x 图像) —— 分离 H1(格式/分布偏移) 与 H3(图像线索)。
    答案固定为同一 key 的答案, 只改"指代方式"和"是否有图"。
P4: vanilla(原始 LLaVA) vs origin(SMU-SFT) 的 IT/PT CE 比 —— 差距是否由 SFT 造成。
P5: 长答案 (bio) 的逐 token 位置 CE —— 差距集中在首 token 还是均匀分布。

CE 口径: 训练侧 collate 构 prompt, labels 掩掉 prompt, 只对答案 token 求 CE。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..unlearn._paired import build_pairs
from .patching import batch_to_device, build_batch


@torch.no_grad()
def ce_tokens(model, items, modality, processor, device, max_length, batch_size):
    out = []
    for b in range(0, len(items), batch_size):
        chunk = items[b:b + batch_size]
        batch = build_batch(chunk, modality, processor, max_length)
        ids, attn, pixel, labels = batch_to_device(batch, modality, device)
        logits = model(input_ids=ids, attention_mask=attn, pixel_values=pixel).logits
        lg = logits[:, :-1].float()
        sl = labels[:, 1:]
        ls = F.cross_entropy(lg.reshape(-1, lg.size(-1)), sl.reshape(-1),
                             reduction="none").view(sl.shape)
        m = sl != -100
        for i in range(sl.size(0)):
            out.append(ls[i][m[i]].detach().cpu())
    return out


def agg_tokens(per_sample):
    ce = torch.cat(per_sample) if per_sample else torch.zeros(1)
    nt = torch.tensor([t.numel() for t in per_sample], dtype=torch.float32)
    tw = float((torch.tensor([float(t.sum()) for t in per_sample])).sum() / nt.sum()) if nt.sum() > 0 else float("nan")
    return {"tw": tw, "mean": float(ce.mean()), "n_tok": int(nt.sum()), "n": len(per_sample)}


def ce_cell(model, items, processor, device, max_length, batch_size):
    mod = "mm" if items and items[0]["image"] is not None else "um"
    return agg_tokens(ce_tokens(model, items, mod, processor, device, max_length, batch_size))


def load_model(path, device):
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    processor.num_additional_image_tokens = 1
    model = LlavaForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(device).eval()
    model.config.use_cache = False
    return model, processor


def load_pairs(split_dir):
    return build_pairs(pd.read_parquet(Path(split_dir) / "forget_5" / "train-00000-of-00001.parquet"))


def p3_factor(model, processor, split_dir, device, max_length, batch_size, key="Description"):
    pairs = [p for p in load_pairs(split_dir) if p["key"] == key]
    cells = {name: [] for name in ("it_img", "it_noimg", "pt_noimg", "pt_img", "mismatch_img")}
    for i, p in enumerate(pairs):
        ans = p["mm_a"]
        other = pairs[(i + 1) % len(pairs)]["image"]
        cells["it_img"].append({"image": p["image"], "question": p["mm_q"], "answer": ans})
        cells["it_noimg"].append({"image": None, "question": p["mm_q"], "answer": ans})
        cells["pt_noimg"].append({"image": None, "question": p["um_q"], "answer": ans})
        cells["pt_img"].append({"image": p["image"], "question": p["um_q"], "answer": ans})
        cells["mismatch_img"].append({"image": other, "question": p["mm_q"], "answer": ans})
    res = {"key": key, "n": len(pairs)}
    for name, items in cells.items():
        res[name] = ce_cell(model, items, processor, device, max_length, batch_size)
    base = res["it_img"]["tw"]
    for name in cells:
        res[name]["ratio_vs_it_img"] = res[name]["tw"] / base if base > 0 else float("inf")
    return res


def p4_compare(vanilla_path, origin_path, split_dir, device, max_length, batch_size):
    out = {}
    for tag, path in (("vanilla", vanilla_path), ("origin", origin_path)):
        model, processor = load_model(path, device)
        pairs = load_pairs(split_dir)
        it = [{"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]} for p in pairs]
        pt = [{"question": p["um_q"], "answer": p["um_a"]} for p in pairs]
        a_it = agg_tokens(ce_tokens(model, it, "mm", processor, device, max_length, batch_size))
        a_pt = agg_tokens(ce_tokens(model, pt, "um", processor, device, max_length, batch_size))
        out[tag] = {"it": a_it, "pt": a_pt,
                    "ratio_tw": a_pt["tw"] / a_it["tw"] if a_it["tw"] > 0 else float("inf"),
                    "ratio_mean": a_pt["mean"] / a_it["mean"] if a_it["mean"] > 0 else float("inf")}
        del model
        torch.cuda.empty_cache()
    return out


def p5_positions(model, processor, split_dir, device, max_length, batch_size, key="Description", nbins=10):
    pairs = [p for p in load_pairs(split_dir) if p["key"] == key]
    it = [{"image": p["image"], "question": p["mm_q"], "answer": p["mm_a"]} for p in pairs]
    pt = [{"question": p["um_q"], "answer": p["um_a"]} for p in pairs]
    tok_it = ce_tokens(model, it, "mm", processor, device, max_length, batch_size)
    tok_pt = ce_tokens(model, pt, "um", processor, device, max_length, batch_size)

    def curve(toks):
        bins = [[] for _ in range(nbins)]
        firsts = []
        for t in toks:
            n = t.numel()
            if n == 0:
                continue
            firsts.append(float(t[0]))
            for j in range(n):
                bins[min(int(j / n * nbins), nbins - 1)].append(float(t[j]))
        return {"bins": [float(torch.tensor(b).mean()) if b else float("nan") for b in bins],
                "first_tok": float(torch.tensor(firsts).mean()),
                "first_tok_median": float(torch.tensor(firsts).median())}

    it_c, pt_c = curve(tok_it), curve(tok_pt)
    return {"key": key, "n": len(pairs), "it": it_c, "pt": pt_c,
            "bin_ratio": [p / i if i and i > 0 else float("inf")
                          for i, p in zip(it_c["bins"], pt_c["bins"])]}


def main():
    ap = argparse.ArgumentParser(description="CE 成因后续验证 P3/P4/P5")
    ap.add_argument("--parts", default="p3,p4,p5")
    ap.add_argument("--base", help="origin (llava_smu_ft)")
    ap.add_argument("--vanilla", help="原始 llava-1.5-7b-hf")
    ap.add_argument("--data_split_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--keys", default="Description,Interest")
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    parts = args.parts.split(",")
    result = {}
    if "p3" in parts or "p5" in parts:
        model, processor = load_model(args.base, args.device)
        if "p3" in parts:
            result["p3_factor"] = {k: p3_factor(model, processor, args.data_split_dir,
                                                args.device, args.max_length, args.batch_size, k)
                                   for k in args.keys.split(",")}
            print("== P3 2x2 (reference x image), answer = key's own answer ==")
            for k, r in result["p3_factor"].items():
                print(f"[{k}] n={r['n']}")
                for name in ("it_img", "it_noimg", "pt_noimg", "pt_img", "mismatch_img"):
                    e = r[name]
                    print(f"  {name:>13}: tw={e['tw']:8.4f} ratio_vs_it_img={e['ratio_vs_it_img']:7.3f} n_tok={e['n_tok']}")
        if "p5" in parts:
            result["p5_positions"] = {k: p5_positions(model, processor, args.data_split_dir,
                                                      args.device, args.max_length, args.batch_size, k)
                                      for k in args.keys.split(",")}
            print("== P5 per-position CE curve ==")
            for k, r in result["p5_positions"].items():
                print(f"[{k}] first_tok it={r['it']['first_tok']:.4f} pt={r['pt']['first_tok']:.4f}")
                print("  it bins: " + " ".join(f"{x:.4f}" for x in r["it"]["bins"]))
                print("  pt bins: " + " ".join(f"{x:.4f}" for x in r["pt"]["bins"]))
                print("  ratio  : " + " ".join(f"{x:.2f}" for x in r["bin_ratio"]))
        del model
        torch.cuda.empty_cache()
    if "p4" in parts:
        result["p4_models"] = p4_compare(args.vanilla, args.base, args.data_split_dir,
                                         args.device, args.max_length, args.batch_size)
        print("== P4 vanilla vs origin ==")
        for tag, r in result["p4_models"].items():
            print(f"{tag:>8}: it_tw={r['it']['tw']:.4f} pt_tw={r['pt']['tw']:.4f} "
                  f"ratio_tw={r['ratio_tw']:.2f} ratio_mean={r['ratio_mean']:.2f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
