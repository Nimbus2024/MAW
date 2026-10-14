"""梯度工具：按模态梯度归一（balance）与逐层/全局范数、余弦诊断。

用于 framework 的 `--grad_mode balance`：把 mm/um 两模态梯度各自归一后再相加，
消除尺度差（P0–P4 的 P1 机制），使 Adam 不会因小尺度模态被淹没而几乎不更新它。
"""
import math
import re

import torch

_LAYER_RE = re.compile(r"layers\.\d+")


def trainable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


def make_groups(module, scope):
    """返回 (params, groups)；groups 是 params 的下标分组（逐层 or 全局）。"""
    params = trainable_params(module)
    if scope == "global":
        return params, [list(range(len(params)))]
    name_to_idx = {}
    for i, (n, p) in enumerate(module.named_parameters()):
        if p.requires_grad:
            name_to_idx[n] = i
    buckets = {}
    for n, idx in name_to_idx.items():
        m = _LAYER_RE.search(n)
        buckets.setdefault(m.group(0) if m else "other", []).append(idx)
    return params, list(buckets.values())


def snapshot(params):
    return [p.grad.detach().clone() if p.grad is not None else None for p in params]


def _sqrt_sum(tensors):
    total = 0.0
    for t in tensors:
        if t is not None:
            total += float(t.detach().float().pow(2).sum())
    return math.sqrt(total)


def global_norm(params):
    return _sqrt_sum([p.grad for p in params])


def stats(params, snap):
    """全局 ‖g_mm‖、‖g_um‖（g_um = p.grad − g_mm）与 cos(g_mm, g_um)。"""
    g_mm, g_um, dot = [], [], 0.0
    for p, s in zip(params, snap):
        if p.grad is None or s is None:
            continue
        u = p.grad.detach() - s
        g_mm.append(s)
        g_um.append(u)
        dot += float((s.float() * u.float()).sum())
    n_mm, n_um = _sqrt_sum(g_mm), _sqrt_sum(g_um)
    cos = dot / (n_mm * n_um) if n_mm > 0 and n_um > 0 else 0.0
    return n_mm, n_um, cos


def write_balanced(params, snap, groups, w_mm, w_um, eps):
    """p.grad = w_mm·(g_mm/‖g_mm‖) + w_um·(g_um/‖g_um‖)，范数按 groups 分组各自计算。"""
    for grp in groups:
        idx = [i for i in grp if params[i].grad is not None and snap[i] is not None]
        if not idx:
            continue
        n_mm = _sqrt_sum([snap[i] for i in idx])
        n_um = _sqrt_sum([params[i].grad.detach() - snap[i] for i in idx])
        d_mm = max(n_mm, eps)
        d_um = max(n_um, eps)
        for i in idx:
            g_mm = snap[i]
            g_um = params[i].grad.detach() - g_mm
            params[i].grad = w_mm * (g_mm / d_mm) + w_um * (g_um / d_um)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr
