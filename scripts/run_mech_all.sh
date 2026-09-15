#!/usr/bin/env bash
# 机制实验总编排 (有卡模式, tmux):
#   E6 alpha 扫描 -> E1 模态隔离 (mm/um) -> E2 ΔW 几何 -> E7 控制器变体 (MAW)
# 用法:
#   tmux new -s mech 'cd .../code && ./scripts/run_mech_all.sh'
#   SKIP_E6=1 ./scripts/run_mech_all.sh        # 跳过某段
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_ROOT}/../results}"
WORK_ROOT="${WORK_ROOT:-$(dirname "${RESULTS_ROOT}")}"
export TMPDIR="${TMPDIR:-${WORK_ROOT}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_ROOT}/.cache}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"
ENV_NAME="${ENV_NAME:-maw}"
LOG_DIR="${RESULTS_ROOT}/_analysis"
LOG="${LOG_DIR}/mech_orchestrator.log"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG}") 2>&1

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV_NAME}"
fi
echo "== orchestrator start $(date) =="
echo "== python: $(command -v python) =="

EPOCHS="${EPOCHS:-3}"
export EPOCHS

if [ "${SKIP_E6:-0}" != "1" ]; then
  echo "== [E6] alpha sweep =="
  ALPHAS="${E6_ALPHAS:-0.0667 0.2 0.3333 1 3 5 15}" "${CODE_ROOT}/scripts/run_mech_sweep.sh"
fi

if [ "${SKIP_E1:-0}" != "1" ]; then
  echo "== [E1] modality-isolated (mm) =="
  ALPHAS=1 MODALITY=mm TAG=mm "${CODE_ROOT}/scripts/run_mech_sweep.sh"
  echo "== [E1] modality-isolated (um) =="
  ALPHAS=1 MODALITY=um TAG=um "${CODE_ROOT}/scripts/run_mech_sweep.sh"
fi

if [ "${SKIP_E2:-0}" != "1" ]; then
  echo "== [E2] adapter geometry (mm vs um) =="
  MM_ADAPTER="$(ls -d "${RESULTS_ROOT}"/simNPO/*-mm/model 2>/dev/null | tail -1 || true)"
  UM_ADAPTER="$(ls -d "${RESULTS_ROOT}"/simNPO/*-um/model 2>/dev/null | tail -1 || true)"
  if [ -n "${MM_ADAPTER}" ] && [ -n "${UM_ADAPTER}" ]; then
    (
      cd "${CODE_ROOT}"
      python -m exp.diagnosis.adapter_geometry \
        --adapter_a "${MM_ADAPTER}" --adapter_b "${UM_ADAPTER}" \
        --label_a mm --label_b um \
        --output "${LOG_DIR}/e2_geometry.json"
    )
  else
    echo "!! 缺少 mm/um adapter, 跳过 E2 (${MM_ADAPTER} | ${UM_ADAPTER})"
  fi
fi

if [ "${SKIP_E7:-0}" != "1" ]; then
  for mode in sigmoid_ema sigmoid_gap fixed; do
    echo "== [E7] MAW gamma_mode=${mode} =="
    extra="--gamma_mode ${mode} --rho 0.8 --lmbda 1.0"
    [ "${mode}" = "fixed" ] && extra="${extra} --gamma_fixed 0.5"
    METHOD=MAW LABEL=MAW NPROC="${MAW_NPROC:-2}" ALPHAS=1 \
      BS="${MAW_BS:-2}" EPOCHS="${MAW_EPOCHS:-3}" LR="${MAW_LR:-5e-6}" \
      TAG="${mode}" EXTRA_ARGS="${extra}" \
      "${CODE_ROOT}/scripts/run_mech_sweep.sh"
  done
fi

echo "== orchestrator done $(date) =="
