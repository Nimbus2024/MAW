# 环境配置说明

本目录存放实验环境的依赖声明与配置入口。首次在（新）服务器上配置环境时，
**按此 README 操作，不要手动逐项安装**，以免与服务器 GPU/CUDA 环境不符。

## 内容

- `requirements.txt` — 项目核心 Python 依赖。

## 关键注意

1. **torch 版本不在此锁定**：由服务器 GPU/CUDA 决定。
   例如 AutoDL RTX 5090 需 `torch>=2.4`（cu128），须在环境里单独安装匹配版本，
   再 `pip install -r requirements.txt`。
2. `vllm` 为 `eval_vllm.py` 的推理后端，版本同样随 CUDA 环境选择（曾用 0.11.0）。
3. 数据与预训练权重**统一放服务器 `dependencies/{data,models}`**，
   由一次性环境配置脚本完成，实验代码只读不重下。

## 待补（服务器环境就绪后完成）

- 服务器环境搭建脚本（`scripts/setup_env*.sh`）：负责
  创建 `dependencies/{data,models}`、按服务器实际 CUDA 装 torch/vllm、
  设置 HF 缓存 / 镜像等环境变量。
- 实验启动脚本对齐项目目录/命名规范（label 白名单、`YYYYMMDD_HHMMSS` 时间戳、
  `logs/stdout.log` 重定向、tmux 运行）。
