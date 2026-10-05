#!/usr/bin/env bash
# =============================================================================
# run_p02_mech.sh — P0/P1/P2: 归一化(clip/Adam)与尺度对跨模态更新主导的作用
# =============================================================================
#   STAGES=refs,p0,p1,p2   选择阶段
#   refs : mm-only / um-only  (ΔW 分解参照)
#   p0   : joint, clip=1.0, --grad_log  (全轨迹梯度记账 + baseline)
#   p1   : joint, --grad_balance        (逐模态梯度归一, 消除尺度差)
#   p2   : joint, clip ∈ ${CLIPS}       (阈值扫描)
# 每个 run 训练后跑 ce_breakdown(--adapter) 记录 IT/PT 问题平均 CE。
# =============================================================================
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-${CODE_ROOT}/../results}"
WORK_ROOT="${WORK_ROOT:-$(dirname "${RESULTS_ROOT}")}"
DEP_ROOT="${DEP_ROOT:-${CODE_ROOT}/../dependencies}"
OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/_analysis}"
GPU="${GPU:-0}"
ENV_NAME="${ENV_NAME:-maw}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
STAGES="${STAGES:-refs,p0,p1,p2}"
CLIPS="${CLIPS:-0.2 5.0 100.0}"

ORIGIN="${DEP_ROOT}/models/llava_smu_ft"
DATA="${DEP_ROOT}/data/UMU-bench"
BS="${BS:-16}"; LR="${LR:-1e-5}"; EP="${EP:-5}"; BETA="${BETA:-0.3}"
R="${R:-8}"; RA="${RA:-16}"

export TMPDIR="${TMPDIR:-${WORK_ROOT}/tmp}"
mkdir -p "${TMPDIR}" "${OUT_ROOT}"
for C in /opt/conda/etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh; do
  if [ -f "$C" ]; then source "$C"; conda activate "${ENV_NAME}"; break; fi
done
cd "${CODE_ROOT}"
COMMON=(--vanilla_dir "${ORIGIN}" --processor_dir "${ORIGIN}" --data_split_dir "${DATA}"
        --forget_split_ratio 5 --batch_size "${BS}" --lr "${LR}" --num_epochs "${EP}"
        --beta "${BETA}" --alpha 1.0 --lora_r "${R}" --lora_alpha "${RA}" --max_length 1024)

run_one() {  # $1=tag $2=modality $3...=extra
  local tag="$1" modality="$2"; shift 2
  local rundir="${OUT_ROOT}/p02_${tag}_${TS}"
  echo "==== [P02] ${tag} modality=${modality} -> ${rundir} ===="
  CUDA_VISIBLE_DEVICES="${GPU}" python -m exp.unlearn.NPO \
    --run_dir "${rundir}" --modality "${modality}" "${COMMON[@]}" "$@" \
    > "${rundir}.train.log" 2>&1 || { echo "TRAIN FAILED ${tag}"; tail -5 "${rundir}.train.log"; return 1; }
  echo "  train done; eval CE"
  CUDA_VISIBLE_DEVICES="${GPU}" python -m exp.diagnosis.ce_breakdown \
    --base "${ORIGIN}" --adapter "${rundir}/model" --data_split_dir "${DATA}" --tasks p1 \
    --output "${OUT_ROOT}/p02_${tag}_${TS}.ce.json" \
    > "${OUT_ROOT}/p02_${tag}_${TS}.ce.log" 2>&1 || echo "  ce_breakdown failed"
  python - "${OUT_ROOT}/p02_${tag}_${TS}.ce.json" <<'PY' 2>/dev/null || true
import json,sys
o=json.load(open(sys.argv[1]))["p1_qa"]["overall"]
print(f"  CE: IT={o['it']['mean']:.5f} PT={o['pt']['mean']:.5f} ratio={o['pt']['mean']/o['it']['mean']:.2f}")
PY
}

mkdir -p "${OUT_ROOT}"
IFS=',' read -ra STAGE_ARR <<< "${STAGES}"
for ST in "${STAGE_ARR[@]}"; do
  case "${ST}" in
    refs) run_one refmm mm ; run_one refum um ;;
    p0)   run_one p0clip1.0 both --clip_max_norm 1.0 --grad_log ;;
    p1)   run_one p1bal both --grad_balance ;;
    p2)   for c in ${CLIPS}; do run_one "clip${c}" both --clip_max_norm "${c}" --grad_log; done ;;
  esac
done
echo "P02 DONE TS=${TS}"
