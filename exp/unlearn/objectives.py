"""可插拔的 LLM unlearn 目标（每个方法一个纯函数 + 冻结声明项）。

框架把方法视为**单模态 forget loss** 的插件：
    L = 2[(1-γ)·L_obj(mm) + γ·L_obj(um)] + λ·KL(π_ref‖π_θ)

统一约定
- 每个函数签名 `fn(model, ref, l, w, args) -> (loss, reward_l, reward_w|None)`
  - `l` = 正确答案(forget)分支 batch；`w` = 拒绝回答(idk)分支 batch（可为 None）
  - `reward_*` = **长度归一**的逐样本 reward（ref-based 方法为 ref 相对量），
    已 detach，仅供控制器计算 `M = reward_w - reward_l`；loss 内部可不用 `w`。
- `needs_ref` / `needs_w` 声明前向依赖：`needs_w=False` 的 objective 在静态 γ 下
  完全不前向 `w`（由 framework 按需构造）。
"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from ._common import _sequence_logprob, forward


@dataclass(frozen=True)
class Objective:
    name: str
    needs_ref: bool
    needs_w: bool
    fn: Callable


def _policy_logp(model, batch, normalize=True):
    return _sequence_logprob(forward(model, batch).logits, batch["labels"], normalize=normalize)


# ── GA: gradient ascent（最大化 forget 答案的 CE） ────────────────────────────
def ga(model, ref, l, w, args):
    lp = _policy_logp(model, l)
    loss = -lp.mean()
    reward_l = lp.detach()
    reward_w = _policy_logp(model, w).detach() if w is not None else None
    return loss, reward_l, reward_w


# ── NPO: ref 相对的负偏好优化 ────────────────────────────────────────────────
def npo(model, ref, l, w, args):
    ratio = _policy_logp(model, l) - _policy_logp(ref, l)
    loss = -(2.0 / args.beta) * F.logsigmoid(-args.beta * ratio - args.obj_gamma).mean()
    reward_l = ratio.detach()
    if w is not None:
        ratio_w = _policy_logp(model, w) - _policy_logp(ref, w)
        reward_w = ratio_w.detach()
    else:
        reward_w = None
    return loss, reward_l, reward_w


# ── simPO: 无参考模型的简化偏好优化 ─────────────────────────────────────────
def simpo(model, ref, l, w, args):
    r_l = _policy_logp(model, l)
    if w is None:
        raise ValueError("simPO objective requires the idk (w) branch")
    r_w = _policy_logp(model, w)
    loss = -(1.0 / args.beta) * F.logsigmoid(args.beta * (r_w - r_l) - args.obj_gamma).mean()
    return loss, r_l.detach(), r_w.detach()


# ── DPO: ref 相对的偏好优化 ──────────────────────────────────────────────────
def dpo(model, ref, l, w, args):
    if w is None:
        raise ValueError("DPO objective requires the idk (w) branch")
    reward_l = _policy_logp(model, l) - _policy_logp(ref, l)
    reward_w = _policy_logp(model, w) - _policy_logp(ref, w)
    loss = -F.logsigmoid(args.beta * (reward_w - reward_l)).mean()
    if args.coeff == "inv_beta":
        loss = loss / args.beta
    return loss, reward_l.detach(), reward_w.detach()


def klmin(model, ref, l, w, args):
    raise NotImplementedError(
        "KLmin 定义待定：KL 在文献中是 retain 正则项（GA+KL），方向/CE-vs-KL 与现行实现不一致。"
        "需读完原论文（Yao et al. 2024 ACL；arXiv:2410.08109 Eq.5）后再落实，"
        "并决定它是 'GA + 统一 λ·KL(ref‖θ)' 的别名还是带自有正则的独立方法。"
    )


OBJECTIVES = {
    "GA": Objective("GA", needs_ref=False, needs_w=False, fn=ga),
    "NPO": Objective("NPO", needs_ref=True, needs_w=False, fn=npo),
    "simPO": Objective("simPO", needs_ref=False, needs_w=True, fn=simpo),
    "DPO": Objective("DPO", needs_ref=True, needs_w=True, fn=dpo),
    "KLmin": Objective("KLmin", needs_ref=True, needs_w=False, fn=klmin),
}


def get_objective(name: str) -> Objective:
    if name not in OBJECTIVES:
        raise KeyError(f"unknown objective {name!r}; choices: {sorted(OBJECTIVES)}")
    return OBJECTIVES[name]
