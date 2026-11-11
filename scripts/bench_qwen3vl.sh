#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STAGE_ROOT="${STAGE_ROOT:-${CODE_ROOT}/../qwen3vl_stage}"
MODEL_DIR="${MODEL_DIR:-${CODE_ROOT}/../dependencies/models}"
DATA_DIR="${DATA_DIR:-${CODE_ROOT}/../dependencies/data/UMU-bench}"
MODEL_ID="${MODEL_ID:-${MODEL_DIR}/Qwen3-VL-8B-Instruct}"
PARQUET="${PARQUET:-${DATA_DIR}/full_data/train-00000-of-00001.parquet}"
PYTHON_STD="${PYTHON_STD:-${CODE_ROOT}/../venv_vllm/bin/python}"
PYTHON_UNSLOTH="${PYTHON_UNSLOTH:-${CODE_ROOT}/../venv_unsloth/bin/python}"
STEPS="${BENCH_STEPS:-20}"
SAMPLES="${BENCH_SAMPLES:-200}"
WORKERS="${BENCH_WORKERS:-8}"

if [[ "${STAGE_ROOT}" != /* ]]; then
  STAGE_ROOT="${CODE_ROOT}/${STAGE_ROOT}"
fi
STAGE_ROOT="$(realpath -m "${STAGE_ROOT}")"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${STAGE_ROOT}/bench/${TS}"
mkdir -p "${OUT_ROOT}"

run_case() {
  local name="$1" python="$2" attn="$3" batch="$4" accum="$5" unsloth="$6"
  local out="${OUT_ROOT}/${name}"
  mkdir -p "${out}"
  local extra=()
  if [[ "${unsloth}" == "1" ]]; then
    extra+=(--use_unsloth)
  else
    extra+=(--no-use_unsloth)
  fi
  echo "== bench ${name} =="
  (
    cd "${CODE_ROOT}"
    "${python}" -m exp.finetune.train_qwen3vl \
      --model_id "${MODEL_ID}" \
      --parquet "${PARQUET}" \
      --output_dir "${out}" \
      --modalities mm \
      --max_samples "${SAMPLES}" \
      --max_steps "${STEPS}" \
      --skip_save \
      --batch_size "${batch}" \
      --grad_accum "${accum}" \
      --num_workers "${WORKERS}" \
      --logging_steps 1 \
      --attn_implementation "${attn}" \
      "${extra[@]}"
  ) > "${out}/stdout.log" 2>&1
  local status=$?
  if [[ -f "${out}/benchmark.json" ]]; then
    echo "== ${name} result =="
    cat "${out}/benchmark.json"
  else
    echo "== ${name} failed (exit ${status}); see ${out}/stdout.log =="
    tail -20 "${out}/stdout.log"
  fi
}

run_case sdpa_b1a8 "${PYTHON_STD}" sdpa 1 8 0
run_case fa2_b1a8 "${PYTHON_STD}" flash_attention_2 1 8 0
run_case sdpa_b4a2 "${PYTHON_STD}" sdpa 4 2 0
run_case fa2_b4a2 "${PYTHON_STD}" flash_attention_2 4 2 0
if [[ -x "${PYTHON_UNSLOTH}" ]]; then
  run_case unsloth_b4a2 "${PYTHON_UNSLOTH}" auto 4 2 1
else
  echo "== skip unsloth: ${PYTHON_UNSLOTH} not executable =="
fi

echo "== summary =="
"${PYTHON_STD}" - "${OUT_ROOT}" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(root, "*", "benchmark.json"))):
    with open(path) as handle:
        data = json.load(handle)
    rows.append(
        (
            os.path.basename(os.path.dirname(path)),
            data.get("attn_implementation"),
            data.get("used_unsloth"),
            data.get("train_samples_per_second"),
            data.get("train_steps_per_second"),
            data.get("peak_memory_gb"),
        )
    )
print("config\tattn\tunsloth\tsamples/s\tsteps/s\tpeak_GB")
for row in rows:
    print("\t".join("" if value is None else str(value) for value in row))
PY
