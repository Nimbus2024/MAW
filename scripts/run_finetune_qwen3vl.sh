#!/usr/bin/env bash
# Qwen3-VL origin SFT launcher. Results stay under STAGE_ROOT (default ../qwen3vl_stage).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
STAGE_ROOT="${STAGE_ROOT:-${CODE_ROOT}/../qwen3vl_stage}"
MODEL_DIR="${MODEL_DIR:-${CODE_ROOT}/../dependencies/models}"
DATA_DIR="${DATA_DIR:-${CODE_ROOT}/../dependencies/data/UMU-bench}"
RUN_NAME="${RUN_NAME:-qwen3_vl_smu_ft}"
MODEL_ID="${MODEL_ID:-${MODEL_DIR}/Qwen3-VL-8B-Instruct}"
PARQUET="${PARQUET:-${DATA_DIR}/full_data/train-00000-of-00001.parquet}"

EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage_root) STAGE_ROOT="$2"; shift 2 ;;
    --run_name|--label) RUN_NAME="$2"; shift 2 ;;
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

echo "== [$(date +%H:%M:%S)] Qwen3-VL origin SFT =="
echo "== run  : ${RUN_DIR}"
echo "== model: ${MODEL_ID}"
echo "== data : ${PARQUET}"
echo "== final: ${FINAL_DIR}"
echo "== log  : ${LOG_DIR}/stdout.log"

cd "${CODE_ROOT}"
CMD=("${PYTHON}" -m exp.finetune.train_qwen3vl \
  --model_id "${MODEL_ID}" \
  --parquet "${PARQUET}" \
  --output_dir "${FINAL_DIR}" \
  --checkpoint_dir "${CKPT_DIR}" \
  --logging_dir "${LOG_DIR}/tensorboard")
if ((${#EXTRA[@]})); then
  CMD+=("${EXTRA[@]}")
fi
"${CMD[@]}"

if [[ -f "${FINAL_DIR}/train_args.json" ]]; then
  cp "${FINAL_DIR}/train_args.json" "${CFG_DIR}/args.json"
fi
echo "== [$(date +%H:%M:%S)] Done. ${RUN_DIR} =="
