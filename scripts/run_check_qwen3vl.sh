#!/usr/bin/env bash
# Quick fit/generalization check for a Qwen3-VL UMU origin.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
if ! command -v "${PYTHON}" >/dev/null 2>&1 && [[ -x "${CODE_ROOT}/../venv_vllm/bin/python" ]]; then
  PYTHON="${CODE_ROOT}/../venv_vllm/bin/python"
fi
STAGE_ROOT="${STAGE_ROOT:-${CODE_ROOT}/../qwen3vl_stage}"
DATA_DIR="${DATA_DIR:-${CODE_ROOT}/../dependencies/data/UMU-bench}"
MODEL="${MODEL:-}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${DATA_DIR}/full_data/train-00000-of-00001.parquet}"
REAL_PARQUET="${REAL_PARQUET:-${DATA_DIR}/real_person/train-00000-of-00001.parquet}"

EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage_root) STAGE_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
if [[ "${STAGE_ROOT}" != /* ]]; then
  STAGE_ROOT="${CODE_ROOT}/${STAGE_ROOT}"
fi
STAGE_ROOT="$(realpath -m "${STAGE_ROOT}")"
if [[ -z "${MODEL}" ]]; then
  MODEL="$(ls -dt "${STAGE_ROOT}"/*/*/model 2>/dev/null | head -n 1 || true)"
fi
if [[ -n "${MODEL}" && "${MODEL}" != /* ]]; then
  MODEL="${CODE_ROOT}/${MODEL}"
fi
if [[ -n "${MODEL}" ]]; then
  MODEL="$(realpath -m "${MODEL}")"
fi
CHECK_DIR="${STAGE_ROOT}/check/${TS}"
OUT="${CHECK_DIR}/result.json"
mkdir -p "${CHECK_DIR}"
exec > >(tee "${CHECK_DIR}/check.log") 2>&1

[[ -d "${MODEL}" ]] || { echo "!! model not found; set MODEL=... or --model" >&2; exit 1; }

echo "== [$(date +%H:%M:%S)] Qwen3-VL origin check =="
echo "== model: ${MODEL}"
echo "== out  : ${OUT}"

cd "${CODE_ROOT}"
CMD=("${PYTHON}" -m exp.finetune.check_learned_qwen3vl \
  --model "${MODEL}" \
  --train_parquet "${TRAIN_PARQUET}" \
  --real_parquet "${REAL_PARQUET}" \
  --out "${OUT}")
if ((${#EXTRA[@]})); then
  CMD+=("${EXTRA[@]}")
fi
"${CMD[@]}"

echo "== [$(date +%H:%M:%S)] Done. ${OUT} =="
