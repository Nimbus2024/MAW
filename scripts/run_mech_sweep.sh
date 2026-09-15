#!/usr/bin/env bash
# 机制实验批量训练+评估+统计 (E1 模态隔离 / E6 alpha 扫描 共用)。
# 每个配置: 训练 simNPO (--grad_log) -> 逐 epoch forget 集评估 (--dump_details)
# -> 实体级 Δ 统计 (exp.diagnosis.stats)。清单写入 results/_analysis/mech_runs.tsv。
#
# 用法 (code/ 内, tmux, 有卡):
#   ./scripts/run_mech_sweep.sh                                  # E6: alpha 网格
#   ALPHAS="1" MODALITY=mm ./scripts/run_mech_sweep.sh           # E1: 仅视觉 forget
#   ALPHAS="1" MODALITY=um ./scripts/run_mech_sweep.sh           # E1: 仅文本 forget
#   ALPHAS="0.0667 1 15" EPOCHS=3 ./scripts/run_mech_sweep.sh
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_ROOT}/../results}"
MODEL_DIR="${MODEL_DIR:-${CODE_ROOT}/../dependencies/models}"
DATA_SPLIT_DIR="${DATA_SPLIT_DIR:-${CODE_ROOT}/../dependencies/data/UMU-bench}"
ORIGIN_DIR="${ORIGIN_DIR:-${MODEL_DIR}/llava_smu_ft}"
VANILLA_DIR="${VANILLA_DIR:-${MODEL_DIR}/llava-1.5-7b-hf}"
TASK_DATA="${DATA_SPLIT_DIR}/full_data/train-00000-of-00001.parquet"
CELEB_DATA="${DATA_SPLIT_DIR}/real_person/train-00000-of-00001.parquet"

ALPHAS="${ALPHAS:-0.0667 0.2 0.3333 1 3 5 15}"
MODALITY="${MODALITY:-both}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-5e-5}"
BS="${BS:-16}"
BETA="${BETA:-0.4}"
LORA_R="${LORA_R:-8}"
LORA_A="${LORA_A:-16}"
TP_SIZE="${TP_SIZE:-2}"
FORGET_RATIO="${FORGET_RATIO:-5}"
EVAL_BATCH="${EVAL_BATCH:-32}"
EVAL_LAST_N="${EVAL_LAST_N:-1}"
TAG="${TAG:-}"

ANALYSIS_DIR="${RESULTS_ROOT}/_analysis"
mkdir -p "${ANALYSIS_DIR}"
RUNS_TSV="${ANALYSIS_DIR}/mech_runs.tsv"
if [ ! -f "${RUNS_TSV}" ]; then
  printf "run_dir\talpha\tmodality\toracle_dir\n" > "${RUNS_TSV}"
fi

eval_pretrain() {
  local model="$1" out="$2" name="$3"
  mkdir -p "${out}"
  (cd "${CODE_ROOT}" && "${PYTHON}" -m exp.eval.eval_vllm \
    --model_id "${model}" --pretrain --processor_path "${ORIGIN_DIR}" \
    --data_split_folder "${DATA_SPLIT_DIR}" --task_data "${TASK_DATA}" \
    --test_data "${TASK_DATA}" --celebrity_data "${CELEB_DATA}" \
    --output_folder "${out}" --output_file "${name}" \
    --forget_ratio "${FORGET_RATIO}" --scopes forget --dump_details \
    --batch_size "${EVAL_BATCH}" --tensor_parallel_size "${TP_SIZE}" --max_model_len 4096) \
    > "${out}/eval.log" 2>&1
}

eval_adapter() {
  local cache="$1" out="$2" name="$3"
  mkdir -p "${out}"
  (cd "${CODE_ROOT}" && "${PYTHON}" -m exp.eval.eval_vllm \
    --model_id "${VANILLA_DIR}" --cache_path "${cache}" --processor_path "${ORIGIN_DIR}" \
    --data_split_folder "${DATA_SPLIT_DIR}" --task_data "${TASK_DATA}" \
    --test_data "${TASK_DATA}" --celebrity_data "${CELEB_DATA}" \
    --output_folder "${out}" --output_file "${name}" \
    --forget_ratio "${FORGET_RATIO}" --scopes forget --dump_details \
    --batch_size "${EVAL_BATCH}" --tensor_parallel_size "${TP_SIZE}" --max_model_len 4096) \
    > "${out}/eval.log" 2>&1
}

if [ -z "${ORACLE_DIR:-}" ]; then
  ORACLE_FILE="${ANALYSIS_DIR}/oracle_dir.txt"
  if [ -f "${ORACLE_FILE}" ]; then
    ORACLE_DIR="$(cat "${ORACLE_FILE}")"
  fi
  if [ -z "${ORACLE_DIR:-}" ] || [ ! -f "${ORACLE_DIR}/forget_fill_details.json" ]; then
    TS_O="$(date +%Y%m%d_%H%M%S)"
    ORACLE_DIR="${RESULTS_ROOT}/origin/${TS_O}/metrics"
    echo "== [$(date +%H:%M:%S)] oracle eval -> ${ORACLE_DIR}"
    eval_pretrain "${ORIGIN_DIR}" "${ORACLE_DIR}" "origin"
    echo "${ORACLE_DIR}" > "${ORACLE_FILE}"
  fi
fi
echo "== oracle: ${ORACLE_DIR}"

for alpha in ${ALPHAS}; do
  TS="$(date +%Y%m%d_%H%M%S)"
  RUN_DIR="${RESULTS_ROOT}/simNPO/${TS}${TAG:+-${TAG}}"
  mkdir -p "${RUN_DIR}/logs/tensorboard" "${RUN_DIR}/config" "${RUN_DIR}/diagnosis"
  echo "== [$(date +%H:%M:%S)] train modality=${MODALITY} alpha=${alpha} -> ${RUN_DIR}"
  (
    cd "${CODE_ROOT}"
    "${PYTHON}" -m exp.unlearn.simNPO \
      --run_dir "${RUN_DIR}" --vanilla_dir "${ORIGIN_DIR}" \
      --processor_dir "${ORIGIN_DIR}" --data_split_dir "${DATA_SPLIT_DIR}" \
      --forget_split_ratio "${FORGET_RATIO}" --batch_size "${BS}" --lr "${LR}" \
      --num_epochs "${EPOCHS}" --beta "${BETA}" --gamma 0.0 --alpha "${alpha}" \
      --lora_r "${LORA_R}" --lora_alpha "${LORA_A}" \
      --modality "${MODALITY}" --grad_log
  ) > "${RUN_DIR}/logs/stdout.log" 2>&1

  mapfile -t EPOCH_MODELS < <(ls -d "${RUN_DIR}"/runs/epoch-*/model 2>/dev/null | sort -V | tail -n "${EVAL_LAST_N}")
  for epoch_model in "${EPOCH_MODELS[@]}"; do
    [ -d "${epoch_model}" ] || continue
    epoch_name="$(basename "$(dirname "${epoch_model}")")"
    METRICS_DIR="${RUN_DIR}/runs/${epoch_name}/metrics"
    echo "== [$(date +%H:%M:%S)] eval ${RUN_DIR##*/} ${epoch_name}"
    eval_adapter "${epoch_model}" "${METRICS_DIR}" "${epoch_name}"
    (
      cd "${CODE_ROOT}"
      "${PYTHON}" -m exp.diagnosis.stats \
        --oracle_dir "${ORACLE_DIR}" --unlearned_dir "${METRICS_DIR}" \
        --label "${epoch_name}" \
        --output "${RUN_DIR}/diagnosis/${epoch_name}_stats.json"
    ) > "${RUN_DIR}/diagnosis/${epoch_name}_stats.log" 2>&1 || true
  done
  printf "%s\t%s\t%s\t%s\n" "${RUN_DIR}" "${alpha}" "${MODALITY}" "${ORACLE_DIR}" >> "${RUNS_TSV}"
  echo "== [$(date +%H:%M:%S)] modality=${MODALITY} alpha=${alpha} done"
done

echo "== [$(date +%H:%M:%S)] done. summary: ${RUNS_TSV}"
