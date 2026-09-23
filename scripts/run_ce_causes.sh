#!/usr/bin/env bash
# =============================================================================
# run_ce_causes.sh — CE 尺度差异成因 P3/P4/P5 (口径 = 训练对齐的问题平均 CE)
# =============================================================================
# P3: 2x2 因子 (参照方式 x 是否有图), 答案固定为该 key 自己的答案
# P4: vanilla(原始 LLaVA) vs origin(SMU-SFT) 的 IT/PT CE 比
# P5: 长答案 (bio) 的逐 token 位置 CE 曲线
#
# 口径说明: simNPO 的 loss 用 _sequence_logprob(normalize=True) -> 逐样本长度归一化
# log-prob, 再对 batch 内样本求平均。故 CE 报告口径 = 逐问题 CE 的问题平均,
# 不是 HF .loss 的 batch 内 token 加权。
# =============================================================================
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_ROOT}/../results}"
WORK_ROOT="${WORK_ROOT:-$(dirname "${RESULTS_ROOT}")}"
DEP_ROOT="${DEP_ROOT:-${CODE_ROOT}/../dependencies}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/_analysis}"
GPU="${GPU:-0}"
KEYS="${KEYS:-Description,Interest}"
ENV_NAME="${ENV_NAME:-maw}"

export TMPDIR="${TMPDIR:-${WORK_ROOT}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_ROOT}/.cache}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${OUT_DIR}"

for C in /opt/conda/etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh; do
  if [ -f "$C" ]; then source "$C"; conda activate "${ENV_NAME}"; break; fi
done
echo "== python: $(command -v python) =="

cd "${CODE_ROOT}"
echo "== P3/P4/P5 -> ${OUT_DIR}/ce_causes.json (GPU ${GPU}) =="
CUDA_VISIBLE_DEVICES="${GPU}" python -m exp.diagnosis.ce_causes \
  --parts p3,p5,p4 \
  --base "${DEP_ROOT}/models/llava_smu_ft" \
  --vanilla "${DEP_ROOT}/models/llava-1.5-7b-hf" \
  --data_split_dir "${DEP_ROOT}/data/UMU-bench" \
  --keys "${KEYS}" \
  --output "${OUT_DIR}/ce_causes.json" \
  > "${OUT_DIR}/ce_causes.log" 2>&1

echo "DONE -> ${OUT_DIR}/ce_causes.json"
