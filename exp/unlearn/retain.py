"""统一 retain 正则项：forward KL(π_ref ‖ π_θ)，仅在答案 token 上取平均。

与旧 `MAW.compute_retain_kl` 完全一致（前向 KL 更适合“保住 retain”）。
"""
import torch
import torch.nn.functional as F

from ._common import forward


def compute_retain_kl(model, ref_model, batch):
    outputs = forward(model, batch)
    with torch.no_grad():
        ref_outputs = forward(ref_model, batch)

    labels = batch["labels"]
    # logits[:, t] predicts labels[:, t + 1]; only assistant-answer positions.
    valid = labels[:, 1:].ne(-100)
    if not valid.any():
        return outputs.logits.sum() * 0.0

    policy_logp = F.log_softmax(outputs.logits[:, :-1].float(), dim=-1)
    ref_logp = F.log_softmax(ref_outputs.logits[:, :-1].float(), dim=-1)
    token_kl = (ref_logp.exp() * (ref_logp - policy_logp)).sum(dim=-1)
    return token_kl.masked_select(valid).mean()
