# MAW — LLM / MLLM Unlearning 实验代码

多模态大模型遗忘（unlearning）研究实验程序：微调、遗忘方法、重训练与评估。

## 结构

```text
.
├── LICENSE
├── env/                     # 环境依赖声明（requirements.txt；无独立文档，见下方「环境配置」）
├── scripts/                 # 批处理脚本：setup_env.sh(建环境) / run_unlearn.sh / run_retrain.sh / run_eval.sh
└── exp/                     # 实验功能程序（Python 包，以 python -m exp.<task>.<mod> 运行）
    ├── finetune/            # 微调（ft_dataset / finetune / info_pre）
    ├── unlearn/             # 遗忘：统一框架 + 可插拔目标
    │                        #   framework.py  唯一训练入口(--objective/--gamma_mode)
    │                        #   objectives.py GA/NPO/simPO/DPO 的 loss(+ KLmin 占位)
    │                        #   controller.py  γ 控制器(static/sigmoid_ema/sigmoid_gap/ema_level)
    │                        #   retain.py 前向 KL(π_ref‖π_θ)；data.py 配对数据
    │                        #   _common.py / _paired.py / unlearn_dataset.py
    ├── retrain/             # 重训练（上界对比，label retrain）
    └── eval/                # 评估（eval_vllm.py 为主；eval_vllm_benchmark.py / eval.py 参考）
```

术语：**vanilla** = 原始基座 `llava-1.5-7b-hf`；**origin** = 微调基线 `llava_smu_ft`
（参考项目称 oracle）。

## 统一框架

所有遗忘方法都是同一个 LLM→MLLM 框架的插件（方法=单模态 loss 函数）：

```
L = 2[(1-γ)·L_obj(mm) + γ·L_obj(um)] + λ·KL(π_ref‖π_θ)
```

- **baseline 组**：静态 γ（默认 0.5，可搜）→ `results/baseline/<algo>/`
- **MAW 组**：动态 γ 控制器 → `results/MAW/<algo>/`
- 目标（`--objective`）：`GA` / `NPO` / `simPO` / `DPO`（`KLmin` 定义待定，调用即报错）
- γ 信号 `M = r_w − r_l`（idk 拒绝 vs 原答案的长度归一 reward，`gap = M_mm − M_um`）；
  **静态 γ 不需要 margin**，框架据此跳过多余的 `w` 分支前向（按需计算）
- retain 统一为前向 `KL(ref‖θ)`，权重 `λ`（默认 1.5，可搜），两组一致

## 环境配置（新服务器一键）

单入口脚本 `scripts/setup_env.sh` 封装完整流程：conda 环境 + pip 镜像 + 依赖 +
torch + 模型/数据集下载（可选 vLLM/验证）。内部委托更合适的工具（conda 建环境、
pip 装库、huggingface_hub 的 `hf` 下载模型与数据）。服务器上**必须用 tmux** 运行。

```bash
tmux new -s setup
cd code
./scripts/setup_env.sh --dry-run          # 先看执行计划（不做事）
./scripts/setup_env.sh --pip-index ustc --yes   # 一键: pip依赖 + 下载模型 + 数据集
```

- 首次运行**自动创建 conda 环境**（默认 `maw`, python 3.10）；`CONDA_CHANNEL` 可切国内
  anaconda 镜像（defaults 在国内常慢）。
- 下载落点：模型 → `../dependencies/models/{llava-1.5-7b-hf, llava_smu_ft}`；
  数据集 UMU-bench → `../dependencies/data/UMU-bench`（代码已接线为本地路径加载）。
- **torch**：有卡时按驱动 CUDA **自动选档**（cu118/121/126/128）并装匹配
  torch/torchvision（PyPI 默认构建，走当前 pip 源避开被墙的 download.pytorch.org）；
  **无卡无法探测 CUDA**，需显式 `--torch cuXXX`，否则装默认最新版。
- **vLLM** 不在基础依赖里：显式 `--vllm [VER]`（auto 命中 cu128 时自动配 0.11.0）
  或 `--all` 才装。
- pip 国内源预设：`--pip-index ustc|aliyun|huawei|tuna|pypi`（或完整 URL），
  写入 `~/.config/pip/pip.conf`。
- 幂等（已完成跳过，`--force` 重下），全程 `tee` 留日志
  `scripts/setup_env_<时间戳>.log`；`--verify` 可对已下模型做加载验证。

已知 CUDA 版本时一次装好 torch + vllm + 依赖 + 下载：

```bash
./scripts/setup_env.sh --pip --torch cu128 --vllm 0.11.0 --models --data --yes
```

## 运行实验 / 评估

```bash
./scripts/run_unlearn.sh GA --eval                 # baseline GA（静态 γ=0.5）
./scripts/run_unlearn.sh MAW/GA --eval             # MAW+GA（动态 γ）
./scripts/run_unlearn.sh MAW/NPO --eval --alpha 0.3 --lmbda 1.5
./scripts/run_unlearn.sh simPO --gamma 0.3 --eval  # 静态 γ 可搜
./scripts/run_retrain.sh --data_dir ...            # retrain 基线
./scripts/run_eval.sh origin --pretrain --model_id <本地模型目录>   # 评估独立模型
```

产物规范：baseline `results/ratio<N>/baseline/<algo>/<timestamp>/`，
MAW `results/ratio<N>/MAW/<algo>/<timestamp>/`；
内含 `logs/{stdout.log,tensorboard/}`、`config/args.json`（含 `group/objective/gamma_mode`）、
`model/`（最终）、`runs/<epoch>/{model,metrics}`（逐 epoch）。

## 约定

- 代码通过 Git 同步到服务器（本地 `git push` → 服务器 `git pull`；**禁止 rsync 同步代码**）；
  实验产物在服务器生成后由本地 rsync 拉回。
- 服务器上运行脚本须用 tmux 防断连。
