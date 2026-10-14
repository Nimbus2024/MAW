#!/usr/bin/env python3
"""X1: 逐参数梯度在 W 的（截断）奇异基下的能量谱 / 支撑重叠 (mm vs um)。

对每个线性层 W (attn & ffn，共 32 层 x 7 投影):
  1) 逐样本的逐参数梯度  gW = dL/dW = sum_{t in answer} delta_t (x) a_t
     （a = 该层输入激活, delta = 该层输出梯度; 只累加答案 token 位置）
  2) 逐模态各自在参考集上估 RMS:  vhat_m[entry] = mean_s (gW_{s,m}[entry])^2
     Adam 式归一:  ghat = gW / (sqrt(vhat_m) + eps)
  3) top-k% 阈值 -> 稀疏 gW_thr (与 W 同形)
  4) 用 W 的截断 SVD 基 (U_K, V_K) 得系数图  C = U_K^T gW_thr V_K
  5) 指标: 能量图 E = |C|^2 的 top-q% Jaccard + Spearman(vec E_mm, vec E_um)
     基线: (a) 随机 top-k% 掩码; (b) 异题同模态 (IT_i vs IT_j)

损失 = 答案 token 的 CE; 模型 = base(origin) 冻结; 仅让输入嵌入输出 require grad 以建图。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..unlearn._paired import build_pairs
from .patching import build_batch, batch_to_device

TARGET = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SKIP = ("multi_modal_projector", "vision_model", "vision_tower", "vision")


def collect_modules(model):
    mods = {}
    for name, m in model.named_modules():
        if any(k in name for k in SKIP):
            continue
        if isinstance(m, torch.nn.Linear) and name.rsplit(".", 1)[-1] in TARGET:
            mods[name] = m
    return mods


class Grabber:
    def __init__(self, modules):
        self.a, self.d, self.h = {}, {}, []
        for name, mod in modules.items():
            self.h.append(mod.register_forward_hook(self._mk_fwd(name)))
            self.h.append(mod.register_full_backward_hook(self._mk_bwd(name)))

    def _mk_fwd(self, name):
        def hook(mod, inp, out):
            self.a[name] = inp[0].detach()
        return hook

    def _mk_bwd(self, name):
        def hook(mod, gin, gout):
            self.d[name] = gout[0].detach()
        return hook

    def clear(self):
        self.a, self.d = {}, {}

    def remove(self):
        for h in self.h:
            h.remove()


def require_input_grad(model):
    emb = model.get_input_embeddings()

    def hook(mod, inp, out):
        out.requires_grad_(True)
        return out

    return emb.register_forward_hook(hook)


def grad_of_W(a, d, pos_mask):
    """a:[1,T,din], d:[1,T,dout], pos_mask:[1,T] -> gW:[dout,din] (float32)."""
    return torch.einsum("bte,btf,bt->ef", d.float(), a.float(), pos_mask.float())


def answer_ce(model, ids, attn, pixel, labels):
    out = model(input_ids=ids, attention_mask=attn, pixel_values=pixel)
    logits = out.logits[:, :-1].float()
    sl = labels[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), sl.reshape(-1),
                           ignore_index=-100)


def answer_pos_mask(labels):
    m = torch.zeros_like(labels)
    m[:, :-1] = labels[:, 1:].ne(-100)
    return m


@torch.enable_grad()
def one_pass(model, grabber, pair, modality, device, processor, max_length):
    item = {"image": pair["image"] if modality == "mm" else None,
            "question": pair["mm_q"] if modality == "mm" else pair["um_q"],
            "answer": pair["mm_a"] if modality == "mm" else pair["um_a"]}
    batch = build_batch([item], modality, processor, max_length)
    ids, attn, pixel, labels = batch_to_device(batch, modality, device)
    grabber.clear()
    model.zero_grad(set_to_none=True)
    loss = answer_ce(model, ids, attn, pixel, labels)
    loss.backward()
    pos = answer_pos_mask(labels)
    out = {n: grad_of_W(grabber.a[n], grabber.d[n], pos).cpu().float().numpy()
           for n in grabber.a if n in grabber.d}
    return out, float(loss.item())


def svd_basis(W, K):
    U, S, V = torch.svd_lowrank(W.float(), q=K)
    return U.cpu(), V.cpu()


def top_q(mask_vals, q_pct):
    n = mask_vals.numel()
    k = max(1, int(round(q_pct / 100.0 * n)))
    thr = torch.topk(mask_vals.reshape(-1), k).values.min()
    return (mask_vals.reshape(-1) >= thr)


def jaccard(x, y):
    i = int((x & y).sum())
    u = int((x | y).sum())
    return i / u if u else float("nan")


def spearman(x, y):
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = rx.norm() * ry.norm()
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data_split_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--forget_split", default="forget_5")
    ap.add_argument("--n_target", type=int, default=3)
    ap.add_argument("--n_ref", type=int, default=50)
    ap.add_argument("--k", type=float, default=1.0, help="top-k%% 阈值")
    ap.add_argument("--q", type=float, default=1.0, help="能量图 top-q%% 用于 Jaccard")
    ap.add_argument("--K", type=int, default=256, help="截断 SVD 秩")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache_basis", default="", help="W 的 SVD 基缓存 .pt 路径")
    args = ap.parse_args()

    torch.set_grad_enabled(True)
    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device).eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    emb_hook = require_input_grad(model)

    modules = collect_modules(model)
    print(f"linear modules: {len(modules)}", flush=True)
    grabber = Grabber(modules)

    df = pd.read_parquet(Path(args.data_split_dir) / args.forget_split
                         / "train-00000-of-00001.parquet")
    pairs = build_pairs(df)
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(args.seed)).tolist()
    targets = [pairs[i] for i in order[:args.n_target]]
    refs = [pairs[i] for i in order[:args.n_ref]]
    print(f"pairs={len(pairs)} targets={len(targets)} refs={len(refs)}", flush=True)

    # ---- phase 1: reference second moment (per modality) ----
    vhat = {m: {n: None for n in modules} for m in ("mm", "um")}
    for m in ("mm", "um"):
        for si, p in enumerate(refs):
            g, _ = one_pass(model, grabber, p, m, args.device, processor, args.max_length)
            for n, w in g.items():
                vhat[m][n] = w * w if vhat[m][n] is None else vhat[m][n] + w * w
            if (si + 1) % 10 == 0:
                print(f"  ref {m} {si+1}/{len(refs)}", flush=True)
        for n in modules:
            vhat[m][n] = np.sqrt(vhat[m][n] / len(refs)) + 1e-8

    # ---- phase 2: target samples (mm / um / cross-question IT) ----
    targets_g = []
    for si, p in enumerate(targets):
        gm, _ = one_pass(model, grabber, p, "mm", args.device, processor, args.max_length)
        gu, _ = one_pass(model, grabber, p, "um", args.device, processor, args.max_length)
        gc = None if si == 0 else targets_g[0]["mm"]  # 异题同模态基线: IT_0 vs IT_i
        targets_g.append({"mm": gm, "um": gu, "cross": gc})
        print(f"  target {si+1}/{len(targets)} done", flush=True)

    # ---- W's truncated SVD basis (cached) ----
    cache_path = Path(args.cache_basis) if args.cache_basis else None
    basis = {}
    if cache_path and cache_path.exists():
        basis = torch.load(cache_path, map_location="cpu")
        print("basis loaded from cache", flush=True)
    else:
        for n, mod in modules.items():
            U, V = svd_basis(mod.weight.detach(), args.K)
            basis[n] = (U, V)
        if cache_path:
            torch.save(basis, cache_path)
    print("basis ready", flush=True)

    # ---- metrics per module ----
    def energy(gW, n, m):
        U, V = basis[n]
        gh = gW / vhat[m][n] if gW is not None else None
        if gh is None:
            return None
        thr = np.partition(np.abs(gh).ravel(), -max(1, int(round(args.k / 100 * gh.size))))[
            -max(1, int(round(args.k / 100 * gh.size)))]
        gthr = np.where(np.abs(gh) >= thr, gh, 0.0)
        C = U.T.numpy() @ gthr @ V.numpy()
        return torch.from_numpy(C * C)  # E = |C|^2

    out = {"config": vars(args), "per_layer": {}, "per_module": {}, "targets": []}
    layer_names = sorted({n.split(".layers.")[1].split(".")[0] for n in modules
                          if ".layers." in n}, key=int)
    for ti, tg in enumerate(targets_g):
        rec = {"index": ti, "modules": {}}
        acc = {}
        for n in modules:
            Emm = energy(tg["mm"][n], n, "mm")
            Eum = energy(tg["um"][n], n, "um")
            Ecr = energy(tg["cross"][n], n, "mm") if tg["cross"] else None
            rand = torch.rand_like(Eum)
            r = {}
            if Emm is not None and Eum is not None:
                bmm, bum = top_q(Emm, args.q), top_q(Eum, args.q)
                r["jaccard_mm_um"] = jaccard(bmm, bum)
                r["spearman_mm_um"] = spearman(Emm.reshape(-1), Eum.reshape(-1))
                branch = top_q(rand, args.q)
                r["jaccard_rand"] = jaccard(bum, branch)
                if Ecr is not None:
                    bcr = top_q(Ecr, args.q)
                    r["jaccard_cross"] = jaccard(bmm, bcr)
                    r["spearman_cross"] = spearman(Emm.reshape(-1), Ecr.reshape(-1))
            rec["modules"][n] = r
            lay = n.split(".layers.")[1].split(".")[0] if ".layers." in n else n
            acc.setdefault(lay, []).append(r)
        # per-layer aggregate (mean over 7 projections)
        for lay, rs in acc.items():
            agg = {}
            for key in ("jaccard_mm_um", "spearman_mm_um", "jaccard_rand",
                        "jaccard_cross", "spearman_cross"):
                vals = [x[key] for x in rs if key in x and not math_isnan(x[key])]
                if vals:
                    agg[key] = float(np.mean(vals))
            out["per_layer"].setdefault(lay, {})[str(ti)] = agg
        out["targets"].append(ti)
        out["per_module"][str(ti)] = rec["modules"]
        print(f"target {ti}: mean J(mm,um)={np.mean([v['jaccard_mm_um'] for v in rec['modules'].values() if 'jaccard_mm_um' in v]):.3f} "
              f"mean rho={np.mean([v['spearman_mm_um'] for v in rec['modules'].values() if 'spearman_mm_um' in v]):.3f} "
              f"| J_cross(IT_i,IT_0)={np.mean([v['jaccard_cross'] for v in rec['modules'].values() if 'jaccard_cross' in v]):.3f}",
              flush=True)

    grabber.remove()
    emb_hook.remove()
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {outp}")


def math_isnan(x):
    try:
        return np.isnan(x)
    except Exception:
        return False


if __name__ == "__main__":
    main()
