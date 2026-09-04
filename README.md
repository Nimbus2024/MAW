# MAW — LLM / MLLM Unlearning 实验代码

基于 `references/unlearning`（Nimbus2024/unlearning 蓝本）迁移重组的实验程序仓库，
目录结构与 AGENTS.md 规范一致。

## 结构

```text
.
├── env/                     # 环境依赖（requirements.txt 等）
├── scripts/                 # 批处理脚本（启动实验/配置环境，.sh）
└── exp/                     # 实验功能程序
    ├── finetune/            # 微调任务（ft_dataset / finetune / info_pre）
    ├── unlearn/             # 遗忘任务（GA / Graddiff / KL / NPO / PO / MAW + unlearn_dataset）
    ├── retrain/             # 重训练任务
    └── eval/                # 评估任务（eval.py / eval_vllm.py / eval_vllm_benchmark.py）
```

## 运行方式

以包模块方式从仓库根目录运行（相对导入依赖此约定）：

```bash
python -m exp.unlearn.NPO    --help
python -m exp.finetune.finetune
python -m exp.eval.eval_vllm
```

## 工作流

- **代码同步**：本地 `git push` → 服务器 `git pull`（禁止 rsync 同步代码）
- **产物同步**：服务器 `results/` → 本地 `product/results/`（仅本地 rsync，方向固定）
- 环境首次配置、实验启动见 `env/` 与 `scripts/` 下的说明（服务器端用 tmux 运行）
