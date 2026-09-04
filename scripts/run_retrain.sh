#!/usr/bin/env bash
# 重训练(retrain)基线启动器 — 产物对齐 AGENTS 规范。
#
# 布局: results/retrain/<timestamp>/{logs/stdout.log, config/args.json, model/}
#
# 用法(在 code/ 内、已激活实验 conda 环境, 建议 tmux):
#   ./scripts/run_retrain.sh --data_dir .../retain_95/train-00000-of-00001.parquet [args...]
#   ./scripts/run_retrain.sh --data_dir <parquet> --num_epochs 3
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_ROOT}/../results}"
DATA_SPLIT_DIR="${DATA_SPLIT_DIR:-${CODE_ROOT}/../dependencies/data/UMU-bench}"

LABEL="retrain"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULTS_ROOT}/${LABEL}/${TS}"
mkdir -p "${RUN_DIR}/logs/tensorboard" "${RUN_DIR}/config"
exec > >(tee "${RUN_DIR}/logs/stdout.log") 2>&1
echo "== [$(date +%H:%M:%S)] ${LABEL} run dir: ${RUN_DIR} =="
echo "== 输出日志: ${RUN_DIR}/logs/stdout.log =="

(
  cd "${CODE_ROOT}"
  "${PYTHON}" -m exp.retrain.retrain --run_dir "${RUN_DIR}" "$@"
)

echo "== [$(date +%H:%M:%S)] Done. ${RUN_DIR} =="
