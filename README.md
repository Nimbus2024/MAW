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
    ├── unlearn/             # 遗忘方法：GA.py / KL.py / MAW.py（对应实验 label GA / KLmin / MAW）
    │                        #   共享 unlearn_dataset.py
    ├── retrain/             # 重训练（上界对比，label retrain）
    └── eval/                # 评估（eval_vllm.py 为主；eval_vllm_benchmark.py / eval.py 参考）
```

术语：**vanilla** = 原始基座 `llava-1.5-7b-hf`；**origin** = 微调基线 `llava_smu_ft`
（参考项目称 oracle）。label `simNPO`/`simPO` 属规划内方法，尚未实现。

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
./scripts/run_unlearn.sh GA --num_epochs 3     # 训练（label GA）
./scripts/run_unlearn.sh MAW --eval            # 训练 + 逐 epoch 评估（GA|KLmin|MAW）
./scripts/run_retrain.sh --data_dir ...        # retrain 基线
./scripts/run_eval.sh origin --pretrain --model_id <本地模型目录>   # 评估独立模型
```

产物规范（`results/<label>/<timestamp>/`）：`logs/{stdout.log,tensorboard/}`、
`config/args.json`、`model/`（最终），MAW 逐 epoch 存 `runs/<epoch>/{model,metrics}`。

## 约定

- 代码通过 Git 同步到服务器（本地 `git push` → 服务器 `git pull`；**禁止 rsync 同步代码**）；
  实验产物在服务器生成后由本地 rsync 拉回。
- 服务器上运行脚本须用 tmux 防断连。
