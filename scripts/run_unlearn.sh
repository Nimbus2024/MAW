#!/usr/bin/env bash
# 遗忘(unlearn)实验启动器 — 统一框架入口。
#
# 布局: <RESULTS_ROOT>/{baseline/<algo>|<algo under MAW>}/<timestamp>/
#         logs/{stdout.log,tensorboard/}, config/args.json,
#         runs/<epoch>/{model,metrics}, model/ (最终)
#
# 两组(统一框架 L = 2[(1-γ)L_obj(mm)+γL_obj(um)] + λ·KL(ref‖θ)):
#   baseline : 静态 γ   ->  results/baseline/<GA|NPO|simPO|DPO|KLmin>/<ts>
#   MAW      : 动态 γ   ->  results/MAW/<GA|NPO|simPO|DPO|KLmin>/<ts>
#
# 用法(在 code/ 内、已激活实验 conda 环境, 建议 tmux 运行):
#   ./scripts/run_unlearn.sh GA --eval                 # baseline GA (静态 γ=0.5)
#   ./scripts/run_unlearn.sh MAW/GA --eval             # MAW+GA (动态 γ)
#   ./scripts/run_unlearn.sh MAW/NPO --eval --alpha 0.3 # 传框架超参
#   ./scripts/run_unlearn.sh simPO --eval --gamma 0.3 --lmbda 1.5
#   MAW_NPROC=4 ./scripts/run_unlearn.sh MAW/DPO --eval
#   RESULTS_ROOT=... DATA_SPLIT_DIR=... ./scripts/run_unlearn.sh GA --eval
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
METHOD="${1:-}"
[ -n "${METHOD}" ] || {
  echo "用法: $0 <GA|NPO|simPO|DPO|KLmin|MAW/GA|MAW/NPO|MAW/simPO|MAW/DPO|MAW/KLmin> [--eval] [framework args...]" >&2
  exit 1
}
shift

DO_EVAL=0
TRAIN_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --eval) DO_EVAL=1 ;;
    *) TRAIN_ARGS+=("${arg}") ;;
  esac
done

# METHOD -> (GROUP, OBJECTIVE, GAMMA_MODE)
case "${METHOD}" in
  GA|NPO|simPO|DPO|KLmin)
    GROUP="baseline"; OBJECTIVE="${METHOD}"; GAMMA_MODE="static" ;;
  MAW/GA|MAW/NPO|MAW/simPO|MAW/DPO|MAW/KLmin)
    GROUP="maw"; OBJECTIVE="${METHOD#MAW/}"; GAMMA_MODE="${MAW_GAMMA_MODE:-ema_level}" ;;
  *)
    echo "不支持的方法: ${METHOD} (合法: GA|NPO|simPO|DPO|KLmin 或 MAW/<algo>)" >&2
    exit 1 ;;
esac
LABEL="${METHOD}"
# 评估输出文件名不能用 '/'，统一换成 '-'
LABEL_FILE="${METHOD//\//-}"

# 每进程数: 默认单进程; 4 卡组用 UNLEARN_NPROC/MAW_NPROC 跑 DDP(大 GBS)
NPROC=1
if [ "${GROUP}" = "maw" ]; then
  [ "${MAW_SINGLE:-0}" = "1" ] && NPROC=1 || NPROC="${MAW_NPROC:-4}"
else
  NPROC="${UNLEARN_NPROC:-1}"
fi
GBS="${GBS:-}"
if [[ -n "${GBS}" ]]; then
  if [ $((GBS % NPROC)) -ne 0 ]; then
    echo "!! GBS=${GBS} 不能被 NPROC=${NPROC} 整除" >&2; exit 1
  fi
  PER_RANK=$((GBS / NPROC))
  TRAIN_ARGS=(--batch_size "${PER_RANK}" "${TRAIN_ARGS[@]}")
  echo "== 全局batch GBS=${GBS} (NPROC=${NPROC}) -> 每进程 batch_size=${PER_RANK} =="
fi

# 依赖路径
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_ROOT}/../results}"
MODEL_DIR="${MODEL_DIR:-${CODE_ROOT}/../dependencies/models}"
DATA_SPLIT_DIR="${DATA_SPLIT_DIR:-${CODE_ROOT}/../dependencies/data/UMU-bench}"
VANILLA_DIR="${VANILLA_DIR:-${MODEL_DIR}/llava-1.5-7b-hf}"
ORIGIN_DIR="${ORIGIN_DIR:-${MODEL_DIR}/llava_smu_ft}"

TS="$(date +%Y%m%d_%H%M%S)${RUN_SUFFIX:-}"
if [ "${GROUP}" = "baseline" ]; then
  RUN_DIR="${RESULTS_ROOT}/baseline/${OBJECTIVE}/${TS}"
else
  RUN_DIR="${RESULTS_ROOT}/MAW/${OBJECTIVE}/${TS}"
fi
mkdir -p "${RUN_DIR}/logs/tensorboard" "${RUN_DIR}/config"
exec > >(tee "${RUN_DIR}/logs/stdout.log") 2>&1
echo "== [$(date +%H:%M:%S)] ${LABEL} (group=${GROUP}, objective=${OBJECTIVE}, gamma=${GAMMA_MODE}) run dir: ${RUN_DIR} =="

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

echo "== 训练: exp.unlearn.framework (objective=${OBJECTIVE}, gamma_mode=${GAMMA_MODE}) =="
FW_ARGS=(--objective "${OBJECTIVE}" --group "${GROUP}" --gamma_mode "${GAMMA_MODE}"
         --run_dir "${RUN_DIR}" --vanilla_dir "${ORIGIN_DIR}"
         --processor_dir "${ORIGIN_DIR}" --data_split_dir "${DATA_SPLIT_DIR}")
# MAW 组默认控制器超参（命令行上的 --alpha/--rho 会覆盖这些默认值）
if [ "${GROUP}" = "maw" ]; then
  FW_ARGS+=(--alpha "${MAW_ALPHA:-0.3}" --rho "${MAW_RHO:-0}")
fi
(
  cd "${CODE_ROOT}"
  if [ "${NPROC}" -gt 1 ]; then
    "${PYTHON}" -m accelerate.commands.launch --num_processes "${NPROC}" \
      -m exp.unlearn.framework "${FW_ARGS[@]}" "${TRAIN_ARGS[@]}"
  else
    "${PYTHON}" -m exp.unlearn.framework "${FW_ARGS[@]}" "${TRAIN_ARGS[@]}"
  fi
)

# 在 args.json 里明确记录 batch 语义
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
  for epoch_model in "${RUN_DIR}"/runs/epoch-*/model; do
    [ -d "${epoch_model}" ] || continue
    local_epoch="$(basename "$(dirname "${epoch_model}")")"
    eval_adapter "${epoch_model}" \
      "${RUN_DIR}/runs/${local_epoch}/metrics" \
      "${LABEL_FILE}_${local_epoch}"
  done
  # 最终 model/ 的全量评估只在 FINAL_FULL=1 时执行
  if [ "${FINAL_FULL:-0}" = "1" ]; then
    EVAL_CAP="" eval_adapter "${RUN_DIR}/model" "${RUN_DIR}/metrics" "${LABEL_FILE}_final"
  fi
fi

echo "== [$(date +%H:%M:%S)] Done. ${RUN_DIR} =="
