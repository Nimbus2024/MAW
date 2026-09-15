#!/usr/bin/env bash
# 双卡并行编排: 每张卡跑一个独立的单卡任务 (规避 DDP/NCCL hang 与 device_map 分片 NaN)。
# E6 剩余 alpha 分两组并行 -> E1 mm/um 并行 -> E2 几何 -> E7 控制器 (2+1 并行)。
# 用法 (tmux): ./scripts/run_mech_parallel.sh
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_ROOT}/../results}"
WORK_ROOT="${WORK_ROOT:-$(dirname "${RESULTS_ROOT}")}"
export TMPDIR="${TMPDIR:-${WORK_ROOT}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_ROOT}/.cache}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"
LOG_DIR="${RESULTS_ROOT}/_analysis"
mkdir -p "${LOG_DIR}"
PYTHON="${PYTHON:-/root/miniconda3/envs/maw/bin/python}"
export PYTHON

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV_NAME:-maw}"
fi
export ORACLE_DIR="${ORACLE_DIR:-$(cat "${LOG_DIR}/oracle_dir.txt" 2>/dev/null || true)}"
echo "== parallel orchestrator start $(date) =="
echo "== oracle: ${ORACLE_DIR:-<will eval>} =="

sweep() {
  local gpu="$1"; shift
  TRAIN_GPU="${gpu}" "${CODE_ROOT}/scripts/run_mech_sweep.sh" "$@"
}

if [ "${SKIP_E6:-0}" != "1" ]; then
  echo "== [E6] remaining alphas (parallel) =="
  sweep 0 "ALPHAS=${E6_GPUS0:-3 5}" TAG=e6a > "${LOG_DIR}/e6_gpu0.log" 2>&1 &
  P0=$!
  sweep 1 "ALPHAS=${E6_GPUS1:-15}" TAG=e6b > "${LOG_DIR}/e6_gpu1.log" 2>&1 &
  P1=$!
  wait "${P0}" || echo "!! E6 gpu0 failed"
  wait "${P1}" || echo "!! E6 gpu1 failed"
fi

if [ "${SKIP_E1:-0}" != "1" ]; then
  echo "== [E1] modality isolation (parallel) =="
  sweep 0 "ALPHAS=1" MODALITY=mm TAG=mm > "${LOG_DIR}/e1_mm.log" 2>&1 &
  P0=$!
  sweep 1 "ALPHAS=1" MODALITY=um TAG=um > "${LOG_DIR}/e1_um.log" 2>&1 &
  P1=$!
  wait "${P0}" || echo "!! E1 mm failed"
  wait "${P1}" || echo "!! E1 um failed"
fi

if [ "${SKIP_E2:-0}" != "1" ]; then
  echo "== [E2] adapter geometry (mm vs um) =="
  MM_ADAPTER="$(ls -d "${RESULTS_ROOT}"/simNPO/*-mm/model 2>/dev/null | tail -1 || true)"
  UM_ADAPTER="$(ls -d "${RESULTS_ROOT}"/simNPO/*-um/model 2>/dev/null | tail -1 || true)"
  if [ -n "${MM_ADAPTER}" ] && [ -n "${UM_ADAPTER}" ]; then
    (cd "${CODE_ROOT}" && "${PYTHON}" -m exp.diagnosis.adapter_geometry \
      --adapter_a "${MM_ADAPTER}" --adapter_b "${UM_ADAPTER}" \
      --label_a mm --label_b um --output "${LOG_DIR}/e2_geometry.json") \
      > "${LOG_DIR}/e2_geometry.log" 2>&1
  else
    echo "!! missing mm/um adapter, skip E2"
  fi
fi

if [ "${SKIP_E7:-0}" != "1" ]; then
  echo "== [E7] MAW controller variants (2 parallel + 1) =="
  sweep 0 "METHOD=MAW" "LABEL=MAW" "ALPHAS=1" "BS=${MAW_BS:-2}" \
    "EPOCHS=${MAW_EPOCHS:-3}" "LR=${MAW_LR:-5e-6}" TAG=sigmoid_ema \
    "EXTRA_ARGS=--gamma_mode sigmoid_ema --rho 0.8 --lmbda 1.0" \
    > "${LOG_DIR}/e7_ema.log" 2>&1 &
  P0=$!
  sweep 1 "METHOD=MAW" "LABEL=MAW" "ALPHAS=1" "BS=${MAW_BS:-2}" \
    "EPOCHS=${MAW_EPOCHS:-3}" "LR=${MAW_LR:-5e-6}" TAG=sigmoid_gap \
    "EXTRA_ARGS=--gamma_mode sigmoid_gap --rho 0.8 --lmbda 1.0" \
    > "${LOG_DIR}/e7_gap.log" 2>&1 &
  P1=$!
  wait "${P0}" || echo "!! E7 ema failed"
  wait "${P1}" || echo "!! E7 gap failed"
  sweep 0 "METHOD=MAW" "LABEL=MAW" "ALPHAS=1" "BS=${MAW_BS:-2}" \
    "EPOCHS=${MAW_EPOCHS:-3}" "LR=${MAW_LR:-5e-6}" TAG=fixed \
    "EXTRA_ARGS=--gamma_mode fixed --gamma_fixed 0.5 --rho 0.8 --lmbda 1.0" \
    > "${LOG_DIR}/e7_fixed.log" 2>&1 || echo "!! E7 fixed failed"
fi

if [ "${SKIP_SUMMARY:-0}" != "1" ]; then
  (cd "${CODE_ROOT}" && "${PYTHON}" -m exp.diagnosis.sweep_summary \
    --runs_tsv "${LOG_DIR}/mech_runs.tsv" \
    --output "${LOG_DIR}/mech_summary.json") > "${LOG_DIR}/summary.log" 2>&1 || true
fi

echo "== parallel orchestrator done $(date) =="
