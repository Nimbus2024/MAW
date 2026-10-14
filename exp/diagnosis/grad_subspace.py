#!/usr/bin/env python3
"""X1: 逐参数梯度在 W 的（截断）奇异基下 的 对角余弦 + 非对角能量占比。

对每个线性层 W (LM 的 attn & ffn，即挂 LoRA 的 224 个矩阵):
  1) 逐样本梯度  gW = dL/dW = (masked delta)^T @ a      (只累加答案 token)
  2) 该模态在参考集上的逐条目 RMS:  vhat[entry] = sqrt(mean_s gW^2) + eps
     Adam 式逐参数归一:  ghat = gW / vhat
  3) top-k% 阈值 -> 稀疏 gW_thr
  4) W 的截断 SVD 基 (U_K,V_K):  C = U_K^T gW_thr V_K   (K x K)
  5) 指标 (逐矩阵 + 逐层聚合):
       cos_diag  = cos( diag(C_mm), diag(C_um) )
       offdiag_m = 1 - ||diag(C_m)||^2 / ||C_m||_F^2
     基线: 异题同模态 cos( diag(C_it_i), diag(C_it_0) )

损失 = 答案 token 的 CE; 模型 = base(origin) 冻结; 仅让输入嵌入输出 require grad 以建图。
张量与 RMS 累加全程留在 GPU（仅最后搬小向量/标量），避免 ~26GB/pass 的 PCIe 拷贝。
"""
from __future__ import annotations

import argparse
import json
import time
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


def _mem(tag):
    if torch.cuda.is_available():
        print(f"  [mem {tag}] alloc={torch.cuda.memory_allocated()/1e9:.1f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.1f}GB", flush=True)


def top_q_np(E, q):
    arr = E.ravel()
    k = max(1, int(round(q / 100.0 * arr.size)))
    thr = np.partition(arr, -k)[-k]
    return arr >= thr


def _jaccard(a, b):
    i = int(np.logical_and(a, b).sum())
    u = int(np.logical_or(a, b).sum())
    return i / u if u else float("nan")


def _spearman_np(x, y):
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / d) if d > 0 else float("nan")


def collect_modules(model):
    mods = {}
    for name, m in model.named_modules():
        if any(k in name for k in SKIP):
            continue
        if isinstance(m, torch.nn.Linear) and name.rsplit(".", 1)[-1] in TARGET:
            mods[name] = m
    return mods


class Grabber:
    """只装 forward hook：存输入激活 a（detach）+ 在输出张量上注册 hook 存 dL/dout（detach）。
    不使用 full_backward_hook（那会额外持有图，导致 pass 后显存不释放）。"""

    def __init__(self, modules):
        self.a, self.d, self.h = {}, {}, []
        for name, mod in modules.items():
            self.h.append(mod.register_forward_hook(self._mk_fwd(name)))

    def _mk_fwd(self, name):
        def hook(mod, inp, out):
            x = inp[0]
            if torch.is_tensor(x):
                self.a[name] = x.detach()
            if torch.is_tensor(out) and out.requires_grad:
                out.register_hook(lambda g, _n=name: self.d.__setitem__(_n, g.detach()))
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
    dm = d.float() * pos_mask.float().unsqueeze(-1)
    return (dm.transpose(1, 2) @ a.float()).squeeze(0)


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
def run_pass(model, grabber, pair, modality, device, processor, max_length, handler, timing=False):
    """一次前向/反向; 逐模块算 gW 后立刻交给 handler(name, gW) 并释放（不整体持有）。"""
    item = {"image": pair["image"] if modality == "mm" else None,
            "question": pair["mm_q"] if modality == "mm" else pair["um_q"],
            "answer": pair["mm_a"] if modality == "mm" else pair["um_a"]}
    batch = build_batch([item], modality, processor, max_length)
    ids, attn, pixel, labels = batch_to_device(batch, modality, device)
    grabber.clear()
    model.zero_grad(set_to_none=True)
    t0 = time.time()
    loss = answer_ce(model, ids, attn, pixel, labels)
    t1 = time.time()
    loss.backward()
    t2 = time.time()
    pos = answer_pos_mask(labels)
    for n in list(grabber.a):
        if n not in grabber.d:
            continue
        gw = grad_of_W(grabber.a[n], grabber.d[n], pos)
        handler(n, gw)
        del gw
    grabber.clear()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if timing:
        print(f"    fwd={t1-t0:.2f} bwd={t2-t1:.2f} dW={time.time()-t2:.2f}", flush=True)
    return float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data_split_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--forget_split", default="forget_5")
    ap.add_argument("--n_target", type=int, default=3)
    ap.add_argument("--n_ref", type=int, default=50)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=768)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache_basis", default="")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--save_C_layers", default="8,16,24,31")
    ap.add_argument("--r_sub", type=int, default=16)
    ap.add_argument("--emap_out", default="")
    args = ap.parse_args()
    torch.set_grad_enabled(True)
    saveC = None if args.save_C_layers.strip().lower() == 'all' else \
        set(x.strip() for x in args.save_C_layers.split(',') if x.strip())

    processor = AutoProcessor.from_pretrained(args.base, local_files_only=True)
    processor.num_additional_image_tokens = 1
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True).to(args.device).eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    emb_hook = require_input_grad(model)

    _mem('model-loaded')
    modules = collect_modules(model)
    names = list(modules)
    print(f"linear modules: {len(names)}", flush=True)
    grabber = Grabber(modules)

    df = pd.read_parquet(Path(args.data_split_dir) / args.forget_split
                         / "train-00000-of-00001.parquet")
    pairs = build_pairs(df)
    order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(args.seed)).tolist()
    targets = [pairs[i] for i in order[:args.n_target]]
    refs = [pairs[i] for i in order[:args.n_ref]]
    print(f"pairs={len(pairs)} targets={len(targets)} refs={len(refs)}", flush=True)

    # W 截断 SVD 基（缓存）
    cache_path = Path(args.cache_basis) if args.cache_basis else None
    if cache_path and cache_path.exists():
        basis = torch.load(cache_path, map_location="cpu")
        print("basis from cache", flush=True)
    else:
        basis = {}
        for n, mod in modules.items():
            U, S, V = torch.svd_lowrank(mod.weight.detach().float(), q=args.K)
            basis[n] = (U.contiguous(), V.contiguous())
        if cache_path:
            torch.save(basis, cache_path)
    _mem('basis')
    print("basis ready", flush=True)

    diags = {}   # diags[m][ti][name] = (diag[K], dnorm2, frob2) on cpu
    for m in ("mm", "um"):
        vhat = {n: torch.zeros(modules[n].weight.shape, dtype=torch.bfloat16,
                               device=args.device) for n in names}
        _mem(f'vhat-{m}')

        def acc_ref(n, gw, _v=vhat):
            _v[n].add_((gw * gw).to(torch.bfloat16))

        for si, p in enumerate(refs):
            run_pass(model, grabber, p, m, args.device, processor,
                     args.max_length, acc_ref, args.timing)
            if si == 0:
                _mem(f'ref-{m}-pass1')
            if (si + 1) % 5 == 0:
                print(f"  ref {m} {si+1}/{len(refs)}", flush=True)
        for n in names:
            vhat[n] = vhat[n].float().sqrt_().div_(np.sqrt(len(refs))).add_(1e-8).to(torch.bfloat16)
        print(f"  vhat[{m}] ready", flush=True)

        diags[m] = []
        for ti, p in enumerate(targets):
            entry = {}

            def acc_target(n, gw, _e=entry, _v=vhat, _m=m, _ti=ti):
                gh = gw / _v[n].float()
                flat = gh.abs().reshape(-1)
                kk = max(1, int(round(args.k / 100.0 * flat.numel())))
                if kk >= flat.numel():          # k=100% -> 不做阈值(去掉稀疏化)
                    gthr = gh
                else:
                    thr = torch.topk(flat, kk).values.min()
                    gthr = torch.where(gh.abs() >= thr, gh, torch.zeros_like(gh))
                U, V = basis[n]
                U = U.to(gthr.device)
                V = V.to(gthr.device)
                C = U.t() @ gthr @ V
                d = torch.diagonal(C)
                rec = {"diag": d.detach().float().cpu(),
                       "dnorm2": float((d * d).sum()),
                       "frob2": float((gthr * gthr).sum()),
                       "shape": (int(gthr.shape[0]), int(gthr.shape[1]))}
                Ug, Sg, Vg = torch.svd_lowrank(gthr, q=args.r_sub)
                rec["Ug"] = Ug.detach().half().cpu()
                rec["Vg"] = Vg.detach().half().cpu()
                lay = n.split(".layers.")[1].split(".")[0] if ".layers." in n else n
                rec["E"] = (C.detach().float() ** 2).half().cpu().numpy()
                rec["lay"] = lay
                if _m == "mm":
                    rec["raw"] = gw.to(torch.bfloat16).cpu()
                else:
                    gm = diags["mm"][_ti][n]["raw"].to(gw.device).float()
                    rec["cos_raw"] = float((gm * gw).sum()
                                           / (gm.norm() * gw.norm() + 1e-12))
                    del gm
                _e[n] = rec

            run_pass(model, grabber, p, m, args.device, processor,
                     args.max_length, acc_target, args.timing)
            diags[m].append(entry)
            print(f"  target {m} {ti+1}/{len(targets)}", flush=True)
        del vhat

    def cosine(a, b):
        d = a.norm() * b.norm()
        return float((a * b).sum() / d) if d > 0 else float("nan")

    out = {"config": vars(args), "per_target": [], "per_layer": {}}
    for ti in range(len(targets)):
        rec = {}
        for n in names:
            em, eu = diags["mm"][ti][n], diags["um"][ti][n]
            r = {"cos_diag": cosine(em["diag"], eu["diag"]),
                 "cos_raw": eu.get("cos_raw"),
                 "offdiag_mm": 1 - em["dnorm2"] / em["frob2"] if em["frob2"] > 0 else None,
                 "offdiag_um": 1 - eu["dnorm2"] / eu["frob2"] if eu["frob2"] > 0 else None}
            Um, Vm = em["Ug"].float(), em["Vg"].float()
            Uu, Vu = eu["Ug"].float(), eu["Vg"].float()
            k = Um.shape[1]
            r["subL_mm_um"] = float((Um.t() @ Uu).pow(2).sum()) / k
            r["subR_mm_um"] = float((Vm.t() @ Vu).pow(2).sum()) / k
            r["subL_rand"] = k / em["shape"][0]
            r["subR_rand"] = k / em["shape"][1]
            Em, Eu = em["E"].astype(np.float32), eu["E"].astype(np.float32)
            r["jaccard_mm_um"] = _jaccard(top_q_np(Em, args.q), top_q_np(Eu, args.q))
            r["spearman_mm_um"] = _spearman_np(Em.ravel(), Eu.ravel())
            if ti > 0:
                e0 = diags["mm"][0][n]
                r["cos_diag_cross_it"] = cosine(em["diag"], e0["diag"])
                U0 = e0["Ug"].float()
                r["subL_it_i0"] = float((Um.t() @ U0).pow(2).sum()) / k
                V0 = e0["Vg"].float()
                r["subR_it_i0"] = float((Vm.t() @ V0).pow(2).sum()) / k
                a0 = e0["raw"].float()
                a1 = em["raw"].float()
                r["cos_raw_cross_it"] = float((a0 * a1).sum()
                                              / (a0.norm() * a1.norm() + 1e-12))
                E0 = e0["E"].astype(np.float32)
                r["jaccard_cross_it"] = _jaccard(top_q_np(Em, args.q), top_q_np(E0, args.q))
                r["spearman_cross_it"] = _spearman_np(Em.ravel(), E0.ravel())
            rec[n] = r
        out["per_target"].append({"index": ti, "modules": rec})
        lay = {}
        for n, r in rec.items():
            L = n.split(".layers.")[1].split(".")[0] if ".layers." in n else n
            lay.setdefault(L, []).append(r)
        for L, rs in lay.items():
            agg = {}
            for key in ("cos_diag", "offdiag_mm", "offdiag_um", "cos_diag_cross_it",
                        "subL_mm_um", "subR_mm_um", "subL_rand", "subR_rand", "subL_it_i0", "subR_it_i0",
                        "jaccard_mm_um", "spearman_mm_um", "jaccard_cross_it", "spearman_cross_it",
                        "cos_raw", "cos_raw_cross_it"):
                vals = [x[key] for x in rs if x.get(key) is not None]
                if vals:
                    agg[key] = float(np.mean(vals))
            out["per_layer"].setdefault(L, {})[str(ti)] = agg
        mc = float(np.mean([r["cos_diag"] for r in rec.values()]))
        mo = float(np.mean([r["offdiag_mm"] for r in rec.values() if r["offdiag_mm"] is not None]))
        sl = np.mean([r["subL_mm_um"] for r in rec.values()])
        sr = np.mean([r["subR_mm_um"] for r in rec.values()])
        rl = np.mean([r["subL_rand"] for r in rec.values()])
        msg = (f"target {ti}: cos_diag(mm,um)={mc:.3f} offdiag_mm={mo:.3f} | "
               f"subL(mm,um)={sl:.3f} (rand {rl:.4f}) subR={sr:.3f}")
        if ti > 0:
            msg += f" | subL(IT_i,IT_0)={np.mean([r['subL_it_i0'] for r in rec.values()]):.3f}"
            msg += f" subR(IT_i,IT_0)={np.mean([r['subR_it_i0'] for r in rec.values()]):.3f}"
        print(msg, flush=True)

    grabber.remove()
    emb_hook.remove()
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.emap_out:
        arrs = {}
        for m in ("mm", "um"):
            for ti in range(len(targets)):
                for n, e in diags[m][ti].items():
                    if saveC is None or e.get("lay") in saveC:
                        arrs[f"{m}|{ti}|{n}"] = e["E"]
        np.savez_compressed(args.emap_out, **arrs)
        print(f"emaps: {args.emap_out} ({len(arrs)} maps)", flush=True)
    print(f"saved: {outp}")


if __name__ == "__main__":
    main()
