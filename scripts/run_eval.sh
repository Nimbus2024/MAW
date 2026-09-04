#!/usr/bin/env bash
# 通用评估启动器 — 对给定模型目录跑 UMU-bench 评估, 产物对齐 AGENTS 规范。
#
# 场景(与 AGENTS eval 存放规则一致):
#   * 独立模型(vanilla/origin):  --pretrain --model_id <本地完整模型目录>
#   * 最终/训练产物(含 adapter):  --cache_path <adapter 或 model 目录> --model_id <基座>
# 结果写入 results/<LABEL>/<timestamp>/metrics/
#
# 用法(在 code/ 内、已激活实验环境, 建议 tmux):
#   ./scripts/run_eval.sh vanilla --pretrain --model_id $MODEL_DIR/llava-1.5-7b-hf
#   ./scripts/run_eval.sh origin  --pretrain --model_id $MODEL_DIR/llava_smu_ft
#   ./scripts/run_eval.sh GA      --cache_path <...>/GA/<ts>/model --model_id $MODEL_DIR/llava-1.5-7b-hf
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_ROOT}/../results}"
DATA_SPLIT_DIR="${DATA_SPLIT_DIR:-${CODE_ROOT}/../dependencies/data/UMU-bench}"
ORIGIN_DIR="${ORIGIN_DIR:-${CODE_ROOT}/../dependencies/models/llava_smu_ft}"

LABEL="${1:-}"
[ -n "${LABEL}" ] || { echo "用法: $0 <LABEL> [eval_vllm args...]" >&2; exit 1; }
shift

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULTS_ROOT}/${LABEL}/${TS}"
METRICS_DIR="${RUN_DIR}/metrics"
mkdir -p "${METRICS_DIR}"
exec > >(tee "${RUN_DIR}/logs/stdout.log") 2>&1
echo "== [$(date +%H:%M:%S)] eval run dir: ${RUN_DIR} =="
echo "== 输出日志: ${RUN_DIR}/logs/stdout.log =="

(
  cd "${CODE_ROOT}"
  "${PYTHON}" -m exp.eval.eval_vllm \
    --processor_path "${ORIGIN_DIR}" \
    --data_split_folder "${DATA_SPLIT_DIR}" \
    --task_data "${DATA_SPLIT_DIR}/full_data/train-00000-of-00001.parquet" \
    --test_data "${DATA_SPLIT_DIR}/full_data/train-00000-of-00001.parquet" \
    --celebrity_data "${DATA_SPLIT_DIR}/real_person/train-00000-of-00001.parquet" \
    --output_folder "${METRICS_DIR}" \
    --output_file "${LABEL}_results" \
    --forget_ratio "${FORGET_RATIO:-5}" \
    --batch_size "${EVAL_BATCH:-32}" --tensor_parallel_size "${TP_SIZE:-4}" \
    --max_model_len 4096 \
    "$@"
)

echo "== [$(date +%H:%M:%S)] Done. ${RUN_DIR} =="
