# 环境配置说明

本目录存放实验环境的依赖声明与**一键配置入口**。

## 新服务器配置环境工作流（一键）

只需在服务器上执行**一个批处理脚本** `scripts/setup_env.sh`，脚本内部按需委托
更合适的工具完成各子步骤（conda 建环境 / pip 装库 / huggingface_hub 的 `hf`
下载模型与数据集），无需记忆多步命令：

```bash
# ①（建议）新开 tmux 会话防断连
tmux new -s setup
cd code

# ② 先看执行计划（不做事）
./scripts/setup_env.sh --dry-run

# ③ 一键执行（默认: pip 依赖 + 下载模型 + 下载数据集）
#    首次运行会自动创建 conda 环境(默认 maw, python 3.10)，一切在该 env 内进行
./scripts/setup_env.sh --pip-index ustc --yes
```

脚本做了什么：

| 子步骤 | 委托工具 | 说明 |
|---|---|---|
| Python 环境 | `conda` | 自动创建 `maw`(python 3.10)；已存在则复用 |
| pip 依赖 | `pip` | `pip install -r env/requirements.txt` |
| torch | `pip` | 可选 `--torch cuXXX`/`auto` 装匹配 CUDA 的版本 |
| **模型下载** | **`hf download`**（Python huggingface_hub CLI） | → `dependencies/models/{llava-1.5-7b-hf, llava_smu_ft}` |
| **数据集下载** | 同上 | → `dependencies/data/UMU-bench` |
| vLLM（可选） | `pip` + import 自检 | 版本随 CUDA 选择（曾用 0.11.0） |

常用选项（全部见 `--help`）：

```bash
./scripts/setup_env.sh --all                          # 默认 + vLLM（vllm 为最新版）
./scripts/setup_env.sh --pip --torch cu128            # 按 cu128 装 torch 再装依赖
# 已知 GPU 的 CUDA 版本时，一次装好匹配的 torch + vllm + 依赖 + 下载：
./scripts/setup_env.sh --pip --torch cu128 --vllm 0.11.0 --models --data --yes
./scripts/setup_env.sh --models --data                # 只下模型/数据集（重下加 --force）
./scripts/setup_env.sh --pip-index huawei             # pip 源预设: ustc|aliyun|huawei|tuna|pypi 或完整 URL
./scripts/setup_env.sh --verify                       # 对已下载模型做 transformers 加载验证
./scripts/setup_env.sh --env-name myenv               # 自定义 conda 环境名
```

要点：全程幂等（已完成自动跳过）；`tee` 留日志
（`scripts/setup_env_<时间戳>.log`）；执行前打印计划表，便于复查问题。

## 术语

- **vanilla** = 原始基座 `llava-hf/llava-1.5-7b-hf`（直接评估，无训练）。
- **origin** = 微调后的基线模型 `llava_smu_ft`（遗忘的上界对比）。
  参考项目中称 **oracle**，本项目统一称 **origin**。

## 关键注意

1. **torch 版本不锁定**：由服务器 GPU/CUDA 决定（如 RTX 5090/Blackwell 需 cu128）。
   `--pip` 不带 `--torch` 时若 torch 缺失，会装默认构建；要精确匹配请用 `--torch cuXXX`。
2. **pip 网络**：部分机房访问 pypi.org / download.pytorch.org 慢或被墙，
   用 `--pip-index` 切国内镜像（实测 USTC、华为云较快）。
3. **HF 下载**：默认走 `hf-mirror.com`，禁用 xet/软链。
4. 数据与权重统一放 `dependencies/{models,data}`，由脚本一次性下载，实验代码只读不重下。

## 目录规范（实验结果侧）

实验代码已接线为从 `dependencies/` 本地路径加载。跑实验/评估见 `scripts/` 下的
`run_unlearn.sh` / `run_retrain.sh` / `run_eval.sh`（产物对齐项目目录/命名规范：
label 白名单、`YYYYMMDD_HHMMSS` 时间戳、`logs/stdout.log`、tmux、runs/）。
