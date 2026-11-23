#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1 && [[ -x "${CODE_ROOT}/../venv_vllm/bin/python" ]]; then
  PYTHON="${CODE_ROOT}/../venv_vllm/bin/python"
fi
STAGE_ROOT="${STAGE_ROOT:-${CODE_ROOT}/../qwen3vl_stage}"
MODEL_DIR="${MODEL_DIR:-${CODE_ROOT}/../dependencies/models}"
DATA_DIR="${DATA_DIR:-${CODE_ROOT}/../dependencies/data/UMU-bench}"
RUN_NAME="${RUN_NAME:-qwen3_vl_hardneg_margin_from_a4}"
MODEL_ID="${MODEL_ID:-${MODEL_DIR}/Qwen3-VL-8B-Instruct}"
PARQUET="${PARQUET:-${DATA_DIR}/full_data/train-00000-of-00001.parquet}"
INIT_ADAPTER="${INIT_ADAPTER:-${STAGE_ROOT}/qwen3_vl_smu_ft_mmum_ep2_ds_b8/20261112_021902/checkpoints/checkpoint-3000}"

EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage_root) STAGE_ROOT="$2"; shift 2 ;;
    --run_name|--label) RUN_NAME="$2"; shift 2 ;;
    --init_adapter) INIT_ADAPTER="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
if [[ "${STAGE_ROOT}" != /* ]]; then
  STAGE_ROOT="${CODE_ROOT}/${STAGE_ROOT}"
fi
STAGE_ROOT="$(realpath -m "${STAGE_ROOT}")"
RUN_DIR="${STAGE_ROOT}/${RUN_NAME}/${TS}"
LOG_DIR="${RUN_DIR}/logs"
CFG_DIR="${RUN_DIR}/config"
FINAL_DIR="${RUN_DIR}/model"
CKPT_DIR="${RUN_DIR}/checkpoints"
mkdir -p "${LOG_DIR}/tensorboard" "${CFG_DIR}" "${FINAL_DIR}" "${CKPT_DIR}"
exec > >(tee "${LOG_DIR}/stdout.log") 2>&1

[[ -d "${MODEL_ID}" ]] || { echo "!! model not found: ${MODEL_ID}" >&2; exit 1; }
[[ -f "${PARQUET}" ]] || { echo "!! parquet not found: ${PARQUET}" >&2; exit 1; }
[[ -d "${INIT_ADAPTER}" ]] || { echo "!! adapter not found: ${INIT_ADAPTER}" >&2; exit 1; }

echo "== [$(date +%H:%M:%S)] Qwen3-VL hard-negative margin =="
echo "== run    : ${RUN_DIR}"
echo "== model  : ${MODEL_ID}"
echo "== adapter: ${INIT_ADAPTER}"
echo "== data   : ${PARQUET}"
echo "== final  : ${FINAL_DIR}"
echo "== log    : ${LOG_DIR}/stdout.log"

cd "${CODE_ROOT}"
CMD=("${PYTHON}" -m exp.finetune.train_qwen3vl_hardneg \
  --model_id "${MODEL_ID}" \
  --parquet "${PARQUET}" \
  --output_dir "${FINAL_DIR}" \
  --checkpoint_dir "${CKPT_DIR}" \
  --logging_dir "${LOG_DIR}/tensorboard" \
  --init_adapter "${INIT_ADAPTER}")
if ((${#EXTRA[@]})); then
  CMD+=("${EXTRA[@]}")
fi
for arg in "${EXTRA[@]}"; do
  if [[ "${arg}" == "--deepspeed" ]]; then
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    export MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}"
    export RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 DS_BUILD_OPS=0
    if [[ "${PYTHON}" == */* ]]; then
      export PATH="$(cd "$(dirname "${PYTHON}")" && pwd):${PATH}"
    fi
    break
  fi
done
"${CMD[@]}"

if [[ -f "${FINAL_DIR}/train_args.json" ]]; then
  cp "${FINAL_DIR}/train_args.json" "${CFG_DIR}/args.json"
fi
echo "== [$(date +%H:%M:%S)] Done. ${RUN_DIR} =="
