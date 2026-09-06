#!/usr/bin/env bash
# 遗忘(unlearn)实验启动器 — 产物严格对齐 AGENTS 规范。
#
# 布局: results/<LABEL>/<timestamp>/{logs/{stdout.log,tensorboard/},
#                                  config/args.json,
#                                  model/                (最终, GA/KLmin/MAW/simNPO/simPO)
#                                  runs/<epoch>/{model/,metrics/}   (逐 epoch: MAW/simNPO/simPO)}
#
# 用法(在 code/ 内、已激活实验 conda 环境, 建议 tmux 运行):
#   ./scripts/run_unlearn.sh GA       --num_epochs 3     # 仅训练
#   ./scripts/run_unlearn.sh KLmin    --eval             # 训练 + 最终评估
#   ./scripts/run_unlearn.sh MAW      --eval             # 训练 + 逐 epoch 评估
#   ./scripts/run_unlearn.sh simNPO   --eval             # simNPO 逐 epoch
#   ./scripts/run_unlearn.sh simPO    --eval             # simPO 逐 epoch
#   MAW_NPROC=4 ./scripts/run_unlearn.sh MAW --eval
#   DATA_SPLIT_DIR=... MODEL_DIR=... ./scripts/run_unlearn.sh GA --eval
#
# 超参规则(第一轮):
#   GBS=<全局batch>  从[64,48,...,2]取首个不OOM; 每进程batch=GBS/NPROC 自动下发
#   EVAL_CAP=<行数>  评估抽样上限(逐 epoch 用, 如 40); 空=全量
#   epoch/lr 依 GBS: GBS>=24 → epoch10/lr1e-4; GBS<24 → epoch5/lr5e-5
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
METHOD="${1:-}"
[ -n "${METHOD}" ] || { echo "用法: $0 <GA|KLmin|MAW|simNPO|simPO> [--eval] [args...]" >&2; exit 1; }
shift

DO_EVAL=0
TRAIN_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --eval) DO_EVAL=1 ;;
    *) TRAIN_ARGS+=("${arg}") ;;
  esac
done

# label↔模块: GA↔GA, KLmin↔KL, MAW/simNPO/simPO↔同名模块
case "${METHOD}" in
  GA)    MODULE="GA"; PER_EPOCH=0 ;;
  KLmin) MODULE="KL"; PER_EPOCH=0 ;;
  MAW|simNPO|simPO) MODULE="${METHOD}"; PER_EPOCH=1 ;;
  *) echo "不支持的方法: ${METHOD} (合法: GA|KLmin|MAW|simNPO|simPO)" >&2; exit 1 ;;
esac
LABEL="${METHOD}"

# 每进程数: MAW 默认 4 卡 DDP, 其余单进程
NPROC=1
[ "${MODULE}" = "MAW" ] && NPROC="${MAW_NPROC:-4}"
GBS="${GBS:-}"
if [[ -n "${GBS}" ]]; then
  if [ $((GBS % NPROC)) -ne 0 ]; then
    echo "!! GBS=${GBS} 不能被 NPROC=${NPROC} 整除" >&2; exit 1
  fi
  PER_RANK=$((GBS / NPROC))
  TRAIN_ARGS=(--batch_size "${PER_RANK}" "${TRAIN_ARGS[@]}")
  echo "== 全局batch GBS=${GBS} (NPROC=${NPROC}) -> 每进程 batch_size=${PER_RANK} =="
fi

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
  local cap=()
  [[ -n "${EVAL_CAP:-}" ]] && cap=(--max_eval_samples "${EVAL_CAP}")
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
    --max_model_len 4096 \
    "${cap[@]}"
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
    # GA/KLmin 无 --processor_dir 参数(用 --model_id 作 processor 源); simNPO/simPO 有
    extra=()
    case "${MODULE}" in simNPO|simPO) extra=(--processor_dir "${ORIGIN_DIR}") ;; esac
    "${PYTHON}" -m "exp.unlearn.${MODULE}" \
      --run_dir "${RUN_DIR}" --vanilla_dir "${ORIGIN_DIR}" \
      --data_split_dir "${DATA_SPLIT_DIR}" \
      "${extra[@]}" \
      "${TRAIN_ARGS[@]}"
  fi
)

# 在 args.json 里明确记录 batch 语义, 避免把每进程当全局
if [ -f "${RUN_DIR}/config/args.json" ]; then
  "${PYTHON}" - "${RUN_DIR}/config/args.json" "${NPROC}" "${GBS:-}" <<'PY'
import json, sys
path, nproc, gbs = sys.argv[1], int(sys.argv[2]), (sys.argv[3] or "")
d = json.load(open(path))
d["num_processes"] = nproc
per = d.get("batch_size")
d["global_batch_size"] = int(gbs) if gbs else (int(per) * nproc if per else None)
json.dump(d, open(path, "w"), indent=2, default=str)
print("== args.json 已记录 num_processes/global_batch_size ==")
PY
fi

if [ "${DO_EVAL}" = "1" ]; then
  if [ "${PER_EPOCH}" = "1" ]; then
    for epoch_model in "${RUN_DIR}"/runs/epoch-*/model; do
      [ -d "${epoch_model}" ] || continue
      local_epoch="$(basename "$(dirname "${epoch_model}")")"
      eval_adapter "${epoch_model}" \
        "${RUN_DIR}/runs/${local_epoch}/metrics" \
        "${LABEL}_${local_epoch}"
    done
    # 最终模型(model/ = 末 epoch 软链)额外做一次全量评估(EVAL_CAP 清空)
    EVAL_CAP="" eval_adapter "${RUN_DIR}/model" "${RUN_DIR}/metrics" "${LABEL}_final"
  else
    eval_adapter "${RUN_DIR}/model" "${RUN_DIR}/metrics" "${LABEL}_final"
  fi
fi

echo "== [$(date +%H:%M:%S)] Done. ${RUN_DIR} =="
