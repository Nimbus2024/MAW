#!/usr/bin/env bash
# =============================================================================
# run_mc_ce.sh — 真·多选题 CE (候选选项间 4 路 softmax), origin base
# =============================================================================
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_ROOT}/../results}"
WORK_ROOT="${WORK_ROOT:-$(dirname "${RESULTS_ROOT}")}"
DEP_ROOT="${DEP_ROOT:-${CODE_ROOT}/../dependencies}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/_analysis}"
GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-maw}"

export TMPDIR="${TMPDIR:-${WORK_ROOT}/tmp}"
mkdir -p "${TMPDIR}" "${OUT_DIR}"

for C in /opt/conda/etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh; do
  if [ -f "$C" ]; then source "$C"; conda activate "${ENV_NAME}"; break; fi
done
echo "== python: $(command -v python) =="

cd "${CODE_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" python -m exp.diagnosis.mc_ce \
  --base "${DEP_ROOT}/models/llava_smu_ft" \
  --data_split_dir "${DEP_ROOT}/data/UMU-bench" \
  --output "${OUT_DIR}/mc_ce.json" \
  > "${OUT_DIR}/mc_ce.log" 2>&1
echo "DONE -> ${OUT_DIR}/mc_ce.json"
