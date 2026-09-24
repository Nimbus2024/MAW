#!/usr/bin/env bash
# =============================================================================
# run_h1_oos.sh — H1 直接检验: PT 格式进入分布后, IT/PT 的 CE 差距是否收窄?
# =============================================================================
# 做法: 从 origin(=llava_smu_ft) 出发, 用普通 LM loss 在 retain_95 上做 1-epoch LoRA SFT
#       (um-only 与 both 两档; 数据用 retain 集, 不碰 forget 实体的知识, 只引入 PT 格式),
#       再用 ce_breakdown.py 评测 forget_5 上的 IT/PT 问题平均 CE 比。
# =============================================================================
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_ROOT}/../results}"
WORK_ROOT="${WORK_ROOT:-$(dirname "${RESULTS_ROOT}")}"
DEP_ROOT="${DEP_ROOT:-${CODE_ROOT}/../dependencies}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/_analysis}"
GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-maw}"
TS="$(date +%Y%m%d_%H%M%S)"

MODELS="${DEP_ROOT}/models"
DATA="${DEP_ROOT}/data/UMU-bench"
ORIGIN="${MODELS}/llava_smu_ft"
RETAIN="${DATA}/retain_95/train-00000-of-00001.parquet"

LR="${LR:-5e-5}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-6}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MODS="${MODS:-um,both}"

export TMPDIR="${TMPDIR:-${WORK_ROOT}/tmp}"
mkdir -p "${TMPDIR}" "${OUT_DIR}"

for C in /opt/conda/etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh; do
  if [ -f "$C" ]; then source "$C"; conda activate "${ENV_NAME}"; break; fi
done
echo "== python: $(command -v python) =="

cd "${CODE_ROOT}"
for TAG in ${MODS//,/ }; do
  case "$TAG" in
    um)   SIDES="um" ;;
    mm)   SIDES="mm" ;;
    both) SIDES="mm,um" ;;
    *) echo "unknown tag $TAG"; exit 1 ;;
  esac
  RUN_DIR="${RESULTS_ROOT}/_analysis/h1_${TAG}_${TS}"
  echo "==== [H1] $TAG (modalities=$SIDES) -> $RUN_DIR ===="
  CUDA_VISIBLE_DEVICES="${GPU}" python -m exp.retrain.retrain \
    --base_model "${ORIGIN}" --processor "${ORIGIN}" \
    --data_dir "${RETAIN}" --run_dir "${RUN_DIR}" \
    --modalities "${SIDES}" --num_epochs "${EPOCHS}" --batch_size "${BATCH}" \
    --lr "${LR}" --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" \
    > "${OUT_DIR}/h1_${TAG}_${TS}.train.log" 2>&1
  echo "==== [H1] $TAG eval CE ===="
  CUDA_VISIBLE_DEVICES="${GPU}" python -m exp.diagnosis.ce_breakdown \
    --base "${ORIGIN}" --adapter "${RUN_DIR}/model" \
    --data_split_dir "${DATA}" --tasks p1 \
    --output "${OUT_DIR}/h1_${TAG}_${TS}.ce.json" \
    > "${OUT_DIR}/h1_${TAG}_${TS}.ce.log" 2>&1
  python - "$OUT_DIR/h1_${TAG}_${TS}.ce.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))["p1_qa"]["overall"]
print(f"  IT={d['it']['mean']:.5f} PT={d['pt']['mean']:.5f} ratio={d['pt']['mean']/d['it']['mean']:.2f}")
PY
done
echo "ALL DONE (TS=${TS})"
