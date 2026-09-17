#!/usr/bin/env python3
"""表示模长 vs 行为: 逐样本相关 + 模长缩放干预 (检验"模长变短→性能更强").

三个检验:
  A. 逐样本相关: 样本在 layer L 的表示模长 vs 该样本答案 CE。
     corr<0 (模长越小 CE 越小) => 支持"短模长更强"; corr>0 => 支持"短模长更弱"。
     同时给 Pearson / Spearman。
  B. 因果干预: 把 layer L 的层输入整体乘 alpha (alpha 扫描), 测平均 CE。
     CE 随 alpha 单调升 => 因果支持"短模长更强"。
  C. 交叉模型: unlearned 模型用 alpha=1/norm_ratio 把模长恢复到 base 水平, 看 CE 是否变差
     (若变差 => base 模长水平更优 => 短模长是损伤; 若变好 => 短模长是增益)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from scipy import stats
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .patching import ce_probes, get_layers, load_pairs, to_device


@torch.no_grad()
def stats_batches(model, probes, modality, layers, device):
    """逐样本: 答案 CE, 每层 norm_last / norm_mean。"""
    ce_all, last, mean = [], {L: [] for L in layers}, {L: [] for L in layers}
    for it in probes:
        ids, attn, pixel, labels = to_device(it["batch"], modality, device)
        o = model(input_ids=ids, attention_mask=attn, pixel_values=pixel,
                  output_hidden_states=True, use_cache=False)
        lg = o.logits[:, :-1].float()
        sl = labels[:, 1:]
        ls = torch.nn.functional.cross_entropy(
            lg.reshape(-1, lg.size(-1)), sl.reshape(-1), reduction="none").view(sl.shape)
        m = sl != -100
        ce_all.append(((ls * m).sum(1) / m.sum(1).clamp_min(1)).cpu())

        mfull = labels != -100
        first = mfull.float().argmax(1)
        pos = (first - 1).clamp_min(0)
        b = torch.arange(ids.size(0), device=device)
        am = attn.unsqueeze(-1).float()
        for L in layers:
            h = o.hidden_states[L].float()
            last[L].append(h[b, pos].norm(dim=1).cpu())
            mean[L].append(((h * am).sum(1) / am.sum(1).clamp_min(1)).norm(dim=1).cpu())
    return {"ce": torch.cat(ce_all),
            "norm_last": {L: torch.cat(v) for L, v in last.items()},
            "norm_mean": {L: torch.cat(v) for L, v in mean.items()}}


@torch.no_grad()
def mean_ce(model, probes, modality, device, hook_layer=None, alpha=1.0):
    h = None
    if hook_layer is not None and alpha != 1.0:
        h = get_layers(model)[hook_layer].register_forward_pre_hook(
            lambda m, a: (a[0] * alpha,))
    try:
        tot, nb = 0.0, 0
        for it in probes:
            ids, attn, pixel, labels = to_device(it["batch"], modality, device)
            out = model(input_ids=ids, attention_mask=attn, pixel_values=pixel, labels=labels)
            tot += float(out.loss)
            nb += 1
        return tot / nb
    finally:
        if h is not None:
            h.remove()


def corr(x, y):
    x, y = x.numpy(), y.numpy()
    return {"pearson": float(stats.pearsonr(x, y)[0]),
            "pearson_p": float(stats.pearsonr(x, y)[1]),
            "spearman": float(stats.spearmanr(x, y)[0]),
            "spearman_p": float(stats.spearmanr(x, y)[1])}


def main():
    p = argparse.ArgumentParser(description="模长 vs 行为")
    p.add_argument("--base", required=True)
    p.add_argument("--adapters", default="", help="name=path,...")
    p.add_argument("--data_split_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--layers", default="16,24,31")
    p.add_argument("--alphas", default="0.6,0.8,1.0,1.2,1.4")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=768)
    p.add_argument("--splits", default="forget_5,retain_shared")
    p.add_argument("--modalities", default="mm,um")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from peft import PeftModel
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    splits = args.splits.split(",")
    modalities = args.modalities.split(",")

    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    base = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device).eval()
    base.config.use_cache = False

    probes = {}
    for sp in splits:
        pairs = load_pairs(args.data_split_dir, sp, args.n, args.seed)
        probes[sp] = ce_probes(pairs, processor, args.max_length, args.batch_size)

    result = {"layers": layers, "alphas": alphas, "n": args.n, "models": {}}

    def run_model(model, name):
        entry = {}
        for sp in splits:
            for mod in modalities:
                st = stats_batches(model, probes[sp], mod, layers, args.device)
                e = {"ce_mean": float(st["ce"].mean()), "n": int(st["ce"].numel()),
                     "corr": {}, "intervention": {}}
                for L in layers:
                    e["corr"][str(L)] = {
                        "norm_last": corr(st["norm_last"][L], st["ce"]),
                        "norm_mean": corr(st["norm_mean"][L], st["ce"]),
                        "norm_last_mean": float(st["norm_last"][L].mean()),
                        "norm_mean_mean": float(st["norm_mean"][L].mean()),
                    }
                    e["intervention"][str(L)] = {
                        str(a): mean_ce(model, probes[sp], mod, args.device, L, a)
                        for a in alphas}
                entry[f"{sp}|{mod}"] = e
                print(f"{name} {sp}|{mod}: ce={e['ce_mean']:.4f}", flush=True)
        print(f"[{name} done]", flush=True)
        return entry

    # base 必须在 PeftModel.from_pretrained 之前跑: PEFT 会原地包装 base,
    # 之后 base 的前向也会带上 adapter。
    result["models"]["base"] = run_model(base, "base")

    adapter_list = [it.split("=", 1) for it in args.adapters.split(",") if it]
    if adapter_list:
        peft = PeftModel.from_pretrained(base, adapter_list[0][1], adapter_name=adapter_list[0][0])
        peft.eval()
        for name, path in adapter_list[1:]:
            peft.load_adapter(path, adapter_name=name)
        for name, _ in adapter_list:
            peft.set_adapter(name)
            result["models"][name] = run_model(peft, name)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, entry in result["models"].items():
        for key, e in entry.items():
            if key.endswith("|um"):
                print(f"{name:>6} {key:>22} ce={e['ce_mean']:.3f} "
                      + " ".join(f"L{L} r={e['corr'][str(L)]['norm_last']['pearson']:+.3f} "
                                 f"sp={e['corr'][str(L)]['norm_last']['spearman']:+.3f}" for L in layers))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
