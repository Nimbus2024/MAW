#!/usr/bin/env bash
# 遗忘(unlearn)实验启动器 — 产物严格对齐 AGENTS 规范。
#
# 布局: results/<LABEL>/<timestamp>/{logs/{stdout.log,tensorboard/},
#                                  config/args.json,
#                                  model/                (最终, GA/KLmin/MAW)
#                                  runs/<epoch>/{model/,metrics/}   (仅 MAW 逐 epoch)}
#
# 用法(在 code/ 内、已激活实验 conda 环境, 建议 tmux 运行):
#   ./scripts/run_unlearn.sh GA       --num_epochs 3     # 仅训练
#   ./scripts/run_unlearn.sh KLmin    --eval             # 训练 + 最终评估
#   ./scripts/run_unlearn.sh MAW      --eval             # 训练 + 逐 epoch 评估
#   MAW_NPROC=4 ./scripts/run_unlearn.sh MAW --eval
#   DATA_SPLIT_DIR=... MODEL_DIR=... ./scripts/run_unlearn.sh GA --eval
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
METHOD="${1:-}"
[ -n "${METHOD}" ] || { echo "用法: $0 <GA|KLmin|MAW> [--eval] [args...]" >&2; exit 1; }
shift

DO_EVAL=0
TRAIN_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --eval) DO_EVAL=1 ;;
    *) TRAIN_ARGS+=("${arg}") ;;
  esac
done

case "${METHOD}" in
  GA)    MODULE="GA" ;;
  KLmin) MODULE="KL" ;;
  MAW)   MODULE="MAW" ;;
  *) echo "不支持的方法: ${METHOD} (合法: GA|KLmin|MAW)" >&2; exit 1 ;;
esac
LABEL="${METHOD}"

# 依赖路径(与 setup_env.sh / exp/_paths.py 一致, 可环境变量覆盖)
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_ROOT}/../results}"
MODEL_DIR="${MODEL_DIR:-${CODE_ROOT}/../dependencies/models}"
DATA_SPLIT_DIR="${DATA_SPLIT_DIR:-${CODE_ROOT}/../dependencies/data/UMU-bench}"
VANILLA_DIR="${VANILLA_DIR:-${MODEL_DIR}/llava-1.5-7b-hf}"
ORIGIN_DIR="${ORIGIN_DIR:-${MODEL_DIR}/llava_smu_ft}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RESULTS_ROOT}/${LABEL}/${TS}"
mkdir -p "${RUN_DIR}/logs/tensorboard" "${RUN_DIR}/config"
exec > >(tee "${RUN_DIR}/logs/stdout.log") 2>&1
echo "== [$(date +%H:%M:%S)] ${LABEL} run dir: ${RUN_DIR} =="
echo "== 输出日志: ${RUN_DIR}/logs/stdout.log =="

eval_adapter() {
  local adapter_dir="$1" out_folder="$2" out_file="$3"
  echo "== [$(date +%H:%M:%S)] Eval ${adapter_dir} -> ${out_folder}/${out_file} =="
  set +e
  "${PYTHON}" -m exp.eval.eval_vllm \
    --model_id "${VANILLA_DIR}" \
    --cache_path "${adapter_dir}" \
    --processor_path "${ORIGIN_DIR}" \
    --data_split_folder "${DATA_SPLIT_DIR}" \
    --task_data "${DATA_SPLIT_DIR}/full_data/train-00000-of-00001.parquet" \
    --test_data "${DATA_SPLIT_DIR}/full_data/train-00000-of-00001.parquet" \
    --celebrity_data "${DATA_SPLIT_DIR}/real_person/train-00000-of-00001.parquet" \
    --output_folder "${out_folder}" \
    --output_file "${out_file}" \
    --forget_ratio "${FORGET_RATIO:-5}" \
    --batch_size "${EVAL_BATCH:-32}" --tensor_parallel_size "${TP_SIZE:-4}" \
    --max_model_len 4096
  local status=$?
  set -e
  if [ "${status}" -ne 0 ]; then
    echo "Eval 失败(status=${status}): ${adapter_dir}" >&2
    exit "${status}"
  fi
}

echo "== 训练: exp.unlearn.${MODULE} =="
(
  cd "${CODE_ROOT}"
  if [ "${MODULE}" = "MAW" ]; then
    "${PYTHON}" -m accelerate.commands.launch --num_processes "${MAW_NPROC:-4}" \
      -m exp.unlearn.MAW \
      --run_dir "${RUN_DIR}" --vanilla_dir "${ORIGIN_DIR}" \
      --processor_dir "${ORIGIN_DIR}" --data_split_dir "${DATA_SPLIT_DIR}" \
      "${TRAIN_ARGS[@]}"
  else
    "${PYTHON}" -m "exp.unlearn.${MODULE}" \
      --run_dir "${RUN_DIR}" --vanilla_dir "${ORIGIN_DIR}" \
      --data_split_dir "${DATA_SPLIT_DIR}" \
      "${TRAIN_ARGS[@]}"
  fi
)

if [ "${DO_EVAL}" = "1" ]; then
  if [ "${MODULE}" = "MAW" ]; then
    for epoch_model in "${RUN_DIR}"/runs/epoch-*/model; do
      [ -d "${epoch_model}" ] || continue
      local_epoch="$(basename "$(dirname "${epoch_model}")")"
      eval_adapter "${epoch_model}" \
        "${RUN_DIR}/runs/${local_epoch}/metrics" \
        "${LABEL}_${local_epoch}"
    done
  else
    eval_adapter "${RUN_DIR}/model" "${RUN_DIR}/metrics" "${LABEL}_final"
  fi
fi

echo "== [$(date +%H:%M:%S)] Done. ${RUN_DIR} =="
