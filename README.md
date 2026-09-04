# MAW — LLM / MLLM Unlearning 实验代码

多模态大模型遗忘（unlearning）研究实验程序：含微调、多类遗忘方法、重训练与评估。

## 结构

```text
.
├── LICENSE
├── env/                     # 环境依赖（requirements.txt + 环境配置说明）
├── scripts/                 # 批处理脚本（启动实验 / 整理结果，.sh）
└── exp/                     # 实验功能程序（Python 包）
    ├── finetune/            # 微调任务（ft_dataset / finetune / info_pre）
    ├── unlearn/             # 遗忘任务（GA / Graddiff / KL / NPO / PO / MAW + unlearn_dataset）
    ├── retrain/             # 重训练任务（上界对比）
    └── eval/                # 评估任务（eval.py / eval_vllm.py / eval_vllm_benchmark.py）
```

## 运行方式

以包模块方式从仓库根目录运行（相对导入依赖此约定，勿直接 `python exp/xxx.py`）：

```bash
python -m exp.unlearn.NPO          --help
python -m exp.finetune.finetune
python -m exp.eval.eval_vllm
```

各遗忘方法（`GA`/`Graddiff`/`KL`/`NPO`/`PO`/`MAW`）共享 `exp/unlearn/unlearn_dataset.py`
中的数据集与 collate 工具。

## 约定

- 环境首次配置、实验启动方式见 `env/` 与 `scripts/` 下的说明。
- 代码通过 Git 同步到执行实验的服务器（`git push` → 服务器 `git pull`），
  产物在服务器生成后由本地 rsync 拉回（禁止用 rsync 同步代码）。
- 服务器上运行实验脚本须用 tmux，防止断连中断任务。
