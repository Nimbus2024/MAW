"""框架公共工具：模型/处理器加载、LoRA 目标、序列 log-prob、随机种子。

这些助手原先散落在 `MAW.py` / `DPO.py` / `simPO.py` 等独立训练程序里；
重构后由 `framework.py` 统一调用，语义保持一致。
"""
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration

from .. import _paths


def load_idk(path="idk.txt"):
    """加载 idk.txt 中的拒绝回答列表。"""
    idk_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    with open(idk_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def find_all_linear_names(model):
    """返回语言模型注意力/MLP 投影层的模块名（排除视觉塔与 projector）。"""
    target_suffixes = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }
    multimodal_keywords = ["multi_modal_projector", "vision_model", "vision_tower"]
    lora_module_names = []
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in target_suffixes:
            lora_module_names.append(name)
    found_suffixes = {name.rsplit(".", 1)[-1] for name in lora_module_names}
    missing = target_suffixes - found_suffixes
    if missing:
        raise ValueError(f"Expected LoRA target modules not found: {sorted(missing)}")
    return sorted(lora_module_names)


def load_model_and_processor(args, load_ref=True):
    """加载 π_θ (SFT + LoRA 前) 与可选的 π_ref (冻结 SFT)。

    load_ref=False 用于不需要参考模型的 objective（GA / simPO）。
    """
    if not _paths.is_llava(args.model_id):
        raise ValueError("Model ID not recognized or not supported.")

    load_kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                       local_files_only=True)
    # Under accelerate launch keep one complete replica per rank (device_map="auto"
    # would split each replica across all GPUs and conflict with DDP).
    if "LOCAL_RANK" in os.environ:
        load_kwargs["device_map"] = {"": int(os.environ["LOCAL_RANK"])}
    else:
        load_kwargs["device_map"] = "auto"

    print("Loading LLAVA policy model (π_θ)...")
    model = LlavaForConditionalGeneration.from_pretrained(args.vanilla_dir, **load_kwargs)
    ref_model = None
    if load_ref:
        print("Loading LLAVA reference model (π_ref, same SFT, frozen)...")
        ref_model = LlavaForConditionalGeneration.from_pretrained(args.vanilla_dir, **load_kwargs)

    proc_dir = args.processor_dir if args.processor_dir else args.model_id
    processor = AutoProcessor.from_pretrained(proc_dir, local_files_only=True)
    # LLaVA expects patch tokens plus the original <image> placeholder (576 for
    # 336x336/14); older processor configs omit it and expand to 575.
    processor.num_additional_image_tokens = 1
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    return model, ref_model, processor


def _sequence_logprob(logits, labels, normalize=False):
    """每个样本一个答案 log-prob（labels 中 prompt/padding 为 -100）。"""
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    valid_counts = valid.sum(dim=-1)
    if (valid_counts == 0).any():
        bad = (valid_counts == 0).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(f"batch contains samples without answer tokens: {bad}")
    safe_targets = targets.masked_fill(~valid, 0)
    token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~valid, 0.0)
    sequence_log_probs = token_log_probs.sum(dim=-1)
    if normalize:
        sequence_log_probs = sequence_log_probs / valid.sum(dim=-1).clamp_min(1)
    return sequence_log_probs


def forward(model, batch):
    """统一的 policy/ref 前向：pixel_values 可为 None（纯文本分支）。"""
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch.get("pixel_values"),
    )


def answer_lengths(batch):
    return batch["labels"][:, 1:].ne(-100).sum(dim=-1).clamp_min(1)


def set_global_seed(seed=42):
    """固定随机种子（不设置 cudnn 标志，保持卷积算法默认选择）。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    seed = 42 + worker_id
    random.seed(seed)
