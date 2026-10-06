#!/usr/bin/env bash
# =============================================================================
# run_p02_eval.sh — 对 P0-P4 的模型跑 UMU-bench eval (串行)
# =============================================================================
#   TAGS="origin refmm refum p0clip1.0 p1bal merged_mm_um" bash scripts/run_p02_eval.sh
# 适配器: base 由 adapter_config.json 的 base_model_name_or_path 决定 (我的都是 origin)。
# 产物: results/_analysis/p02_eval_<tag>/<tag>_final_evaluation_results.json (+ details)
# =============================================================================
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_ROOT}/../results}"
DEP_ROOT="${DEP_ROOT:-${CODE_ROOT}/../dependencies}"
ORIGIN="${DEP_ROOT}/models/llava_smu_ft"
DATA="${DEP_ROOT}/data/UMU-bench"
OUT="${RESULTS_ROOT}/_analysis"
ENV_NAME="${ENV_NAME:-maw}"
TS="${TS:-20261005_204029}"
TAGS="${TAGS:-origin refmm refum p0clip1.0 p1bal merged_mm_um}"
EVAL_BATCH="${EVAL_BATCH:-32}"
GPU="${GPU:-0}"

export TMPDIR="${TMPDIR:-${CODE_ROOT}/../tmp}"
mkdir -p "${TMPDIR}"
for C in /opt/conda/etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh; do
  if [ -f "$C" ]; then source "$C"; conda activate "${ENV_NAME}"; break; fi
done
cd "${CODE_ROOT}"

for T in ${TAGS}; do
  D="${OUT}/p02_eval_${T}"
  mkdir -p "${D}"
  case "${T}" in
    origin)      ARGS=(--pretrain --model_id "${ORIGIN}") ;;
    merged_mm_um) ARGS=(--pretrain --model_id "${OUT}/p02_merged_mm_um/model_full") ;;
    *)           ARGS=(--model_id "${ORIGIN}" --cache_path "${OUT}/p02_${T}_${TS}/model") ;;
  esac
  echo "==== [$(date +%H:%M:%S)] eval ${T} ===="
  CUDA_VISIBLE_DEVICES="${GPU}" python -m exp.eval.eval_vllm "${ARGS[@]}" \
    --processor_path "${ORIGIN}" \
    --data_split_folder "${DATA}" \
    --task_data "${DATA}/full_data/train-00000-of-00001.parquet" \
    --test_data "${DATA}/full_data/train-00000-of-00001.parquet" \
    --celebrity_data "${DATA}/real_person/train-00000-of-00001.parquet" \
    --output_folder "${D}" --output_file "${T}" \
    --forget_ratio 5 --batch_size "${EVAL_BATCH}" --tensor_parallel_size 1 --max_model_len 4096 \
    --scopes forget,retain_shared,retain_celebrity --dump_details \
    > "${D}/eval.log" 2>&1 || echo "!! EVAL FAILED: ${T}"
  echo "==== [$(date +%H:%M:%S)] done ${T} ===="
done
echo "ALL EVALS DONE"
