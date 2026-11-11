# vendored Qwen3-VL finetune logic

Source: <https://github.com/QwenLM/Qwen3-VL>, directory `qwen-vl-finetune/`, Apache-2.0.

Files here are trimmed copies/adaptations of:

- `qwenvl/data/data_processor.py` → `processor.py`, `collator.py`, `datasets.py`
- `qwenvl/data/rope2d.py` → `rope2d.py`

Kept behavior: Qwen3-VL chat template, assistant-only label masking, 3D mRoPE
position ids, standard-padding collation. The upstream flash-attention
monkey patches and video packing paths are intentionally omitted.
