#!/usr/bin/env bash
# =============================================================================
# run_patching_ce.sh — Activation patching (CE 判据) 全量重跑
# =============================================================================
# 口径 = 训练对齐的问题平均 CE (逐样本 CE -> 对样本平均; commit 057571e)。
# 旧结果是 HF .loss 的 batch 内 token 加权 + n=16, 不可用于跨模态幅度论述。
# =============================================================================
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_ROOT}/../results}"
WORK_ROOT="${WORK_ROOT:-$(dirname "${RESULTS_ROOT}")}"
DEP_ROOT="${DEP_ROOT:-${CODE_ROOT}/../dependencies}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/_analysis}"
GPU="${GPU:-0}"
N="${N:-300}"
BATCH="${BATCH:-8}"
LAYERS="${LAYERS:-0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31}"
ENV_NAME="${ENV_NAME:-maw}"

export TMPDIR="${TMPDIR:-${WORK_ROOT}/tmp}"
mkdir -p "${TMPDIR}" "${OUT_DIR}"

for C in /opt/conda/etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh; do
  if [ -f "$C" ]; then source "$C"; conda activate "${ENV_NAME}"; break; fi
done
echo "== python: $(command -v python) =="

cd "${CODE_ROOT}"
echo "== patching CE n=${N} batch=${BATCH} -> ${OUT_DIR}/patching_ce_full.json =="
CUDA_VISIBLE_DEVICES="${GPU}" python -m exp.diagnosis.patching \
  --base "${DEP_ROOT}/models/llava_smu_ft" \
  --adapters "mm=${RESULTS_ROOT}/simNPO/20260915_141455-mm/model,um=${RESULTS_ROOT}/simNPO/20260915_142326-um/model,joint=${RESULTS_ROOT}/simNPO/20260915_133541/model" \
  --data_split_dir "${DEP_ROOT}/data/UMU-bench" \
  --output "${OUT_DIR}/patching_ce_full.json" \
  --metric ce --n "${N}" --batch_size "${BATCH}" --layers "${LAYERS}" \
  > "${OUT_DIR}/patching_ce_full.log" 2>&1

echo "DONE -> ${OUT_DIR}/patching_ce_full.json"
