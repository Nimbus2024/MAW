# 环境配置说明

本目录存放实验环境的依赖声明与配置入口。

## 一键配置（推荐）

首次在（新）服务器上配置环境，请运行一键脚本 `scripts/setup_env.sh`
（服务器上**必须用 tmux** 启动，防止长时间下载被断连中断）：

```bash
tmux new -s setup                          # 新开 tmux 会话
cd code
./scripts/setup_env.sh --dry-run           # ① 先看执行计划（不做事）
./scripts/setup_env.sh --yes               # ② 默认执行: pip 依赖 + 模型 + 数据集
```

脚本负责：安装 `env/requirements.txt` 依赖、把 vanilla / origin 模型下载到
`dependencies/models/`、把 UMU-bench 数据集下载到 `dependencies/data/`；
幂等（已完成自动跳过）、全程 `tee` 留日志
（`scripts/setup_env_<时间戳>.log`），便于复查。更多选项见 `--help`：

```bash
./scripts/setup_env.sh --all               # 全部 + vLLM
./scripts/setup_env.sh --vllm 0.11.0       # 只装指定版本 vLLM
./scripts/setup_env.sh --pip --torch cu128 # 装依赖并按 cu128 索引装 torch
./scripts/setup_env.sh --models --data     # 只重新下模型和数据集
```

> 建议先按本文件「关键注意」手动装好与 GPU 匹配的 torch，再跑一键脚本。

## 术语

- **vanilla** = 原始基座 `llava-hf/llava-1.5-7b-hf`（直接评估，无训练）。
- **origin** = 微调后的基线模型 `llava_smu_ft`（遗忘的上界对比）。
  参考项目中称 **oracle**，本项目统一称 **origin**。

## 关键注意

1. **torch 版本不锁定**：由服务器 GPU/CUDA 决定。例如 AutoDL RTX 5090
   需 `torch>=2.4`（cu128），须在环境里单独安装匹配版本再 `--pip`。
2. `vllm` 为 `eval_vllm.py` 的推理后端，版本随 CUDA 环境选择（曾用 0.11.0）。
3. 数据与权重统一放 `dependencies/{models,data}`，由脚本一次性下载，
   实验代码只读不重下。

## 待补（Phase B，接线）

- 实验代码目前仍按 `repo_id + local_files_only`（命中 HF 缓存）加载；
  待改为从 `dependencies/models/` 本地路径加载（角色 vanilla / origin），
  并重写 `scripts/run_unlearn.sh` 对齐项目目录/命名规范
  （label 白名单、`YYYYMMDD_HHMMSS` 时间戳、`logs/stdout.log`、tmux、runs/）。
