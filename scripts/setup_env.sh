#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — 新服务器一键环境配置：pip 依赖 + 下载模型/数据集（可复查、可复用）
#
# 说明:
#   1) 模型按角色命名落到 dependencies/models/<name>/，数据集落到
#      dependencies/data/UMU-bench/，严格对齐 AGENTS.md 的 dependencies 布局。
#   2) 术语: 参考项目中称 "oracle" 的微调基线，本项目统一称 "origin"。
#        - vanilla = llava-hf/llava-1.5-7b-hf      (原始基座, 直接评估)
#        - origin  = <llava_smu_ft repo>            (微调后的基线模型, 参考项目叫 oracle)
#   3) 全程 xet/软链禁用 + 镜像下载，幂等（已完成则跳过，--force 重下）。
#   4) 用法:
#        ./scripts/setup_env.sh                     # 默认: pip + models + data
#        ./scripts/setup_env.sh --all               # 再加 vLLM
#        ./scripts/setup_env.sh --models --data     # 只下模型和数据集
#        ./scripts/setup_env.sh --pip --torch cu128 # 装依赖并按 cu128 装 torch
#        ./scripts/setup_env.sh --dry-run           # 只打印执行计划，不做事
#     服务器上必须用 tmux 运行（长时间下载，防断连中断）。
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 配置区 —— 均可用环境变量覆盖
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DEP_ROOT="${DEP_ROOT:-${CODE_ROOT}/../dependencies}"      # 服务器工作根下 dependencies/
MODEL_DIR="${MODEL_DIR:-${DEP_ROOT}/models}"
DATA_DIR="${DATA_DIR:-${DEP_ROOT}/data}"

# HF 镜像与传输开关
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_SYMLINKS="${HF_HUB_DISABLE_SYMLINKS:-1}"

# 模型/数据集源前缀: chengyewang = 镜像(蓝本验证过); linbojunzi = 上游官方
HF_SOURCE="${HF_SOURCE:-chengyewang}"

# vanilla 基座 (角色 vanilla)
VANILLA_REPO="${VANILLA_REPO:-llava-hf/llava-1.5-7b-hf}"
VANILLA_NAME="llava-1.5-7b-hf"

# 微调基线: 参考项目叫 oracle，本项目叫 origin
ORIGIN_REPO="${ORIGIN_REPO:-${HF_SOURCE}/llava_smu_ft}"
ORIGIN_NAME="llava_smu_ft"

# 数据集 (完整 UMU-bench, 含 full_data/real_person/forget_5/retain_95 等)
DATA_REPO="${DATA_REPO:-${HF_SOURCE}/UMU-bench}"
DATA_NAME="UMU-bench"

VLLM_VERSION="${VLLM_VERSION:-}"                            # 空 = 最新稳定版(曾用 0.11.0)

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
DO_PIP=0 DO_MODELS=0 DO_DATA=0 DO_VLLM=0
SPECIFIED=0
FORCE=0 DRY_RUN=0 YES=0
TORCH_INDEX=""
TS="$(date +%Y%m%d_%H%M%S)"

usage() {
  sed -n '3,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo
  echo "选项:"
  echo "  --pip            安装依赖 (pip install -r env/requirements.txt)"
  echo "  --torch [cu]     安装匹配 torch/torchvision (cu118|cu121|cu126|cu128|auto)"
  echo "  --models         下载模型到 ${MODEL_DIR}"
  echo "  --data           下载数据集到 ${DATA_DIR}"
  echo "  --vllm [VER]     安装 vLLM (默认最新, 曾用 0.11.0)"
  echo "  --all            等价 --pip --models --data --vllm"
  echo "  --force          已存在也重新下载/安装"
  echo "  --dry-run        只打印执行计划, 不执行"
  echo "  --yes            跳过确认"
  exit 0
}

# ---------------------------------------------------------------------------
# 预检 / 工具
# ---------------------------------------------------------------------------
detect_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    CUDA_VER="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$' || true)"
  fi
  GPU_NAME="${GPU_NAME:-unknown}"; CUDA_VER="${CUDA_VER:-unknown}"
}

map_cuda_to_index() {
  # auto: 由驱动 CUDA 推断; 否则用显式 cuXXX
  case "${1:-}" in
    cu118) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
    cu121) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    cu126) TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
    cu128) TORCH_INDEX="https://download.pytorch.org/whl/cu128" ;;
    auto)
      if [[ "$CUDA_VER" =~ ^([0-9]+)\.([0-9]+) ]]; then
        major=${BASH_REMATCH[1]}; minor=${BASH_REMATCH[2]}
        if   (( major > 12 || (major == 12 && minor >= 8) )); then TORCH_INDEX="https://download.pytorch.org/whl/cu128"
        elif (( major == 12 && minor >= 6 )); then TORCH_INDEX="https://download.pytorch.org/whl/cu126"
        elif (( major >= 12 )); then TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        else TORCH_INDEX="https://download.pytorch.org/whl/cu118"; fi
      else
        echo "!! 无法识别驱动 CUDA, 无法自动选 torch 索引" >&2
        TORCH_INDEX=""
      fi
      ;;
    "") TORCH_INDEX="" ;;
    *) echo "!! 未知 torch 索引: $1 (cu118|cu121|cu126|cu128|auto)" >&2; usage ;;
  esac
}

ensure_hf_cli() {
  if ! command -v hf >/dev/null 2>&1; then
    echo "== huggingface_hub CLI 缺失, 先安装 =="
    pip install -q -U "huggingface_hub[cli]"
  fi
}

# hf download 封装: 幂等按目录内容判断
dl_hf() { # $1=repo  $2=repo-type  $3=目标目录
  local repo="$1" rtype="$2" dst="$3"
  local marker="${dst}/.setup-complete"
  if [[ -f "$marker" ]] && [[ "$FORCE" != 1 ]]; then
    echo "== 已存在(${dst}), 跳过 (--force 重下) =="
    return 0
  fi
  if [[ "$FORCE" == 1 ]] && [[ -e "$dst" ]]; then
    echo "== --force: 清理 ${dst} =="
    rm -rf "$dst"
  fi
  mkdir -p "$dst"
  echo "== hf download ${repo} (${rtype}) -> ${dst} =="
  hf download "$repo" --repo-type "$rtype" --local-dir "$dst"
  touch "$marker"
  echo "== 完成: ${repo} =="
}

# ---------------------------------------------------------------------------
# 各步骤
# ---------------------------------------------------------------------------
step_pip() {
  if [[ -n "$TORCH_INDEX" ]]; then
    echo "== pip install torch/torchvision (index: ${TORCH_INDEX}) =="
    pip install --index-url "$TORCH_INDEX" torch torchvision
  else
    if python -c 'import torch' >/dev/null 2>&1; then
      echo "== torch 已存在, 跳过 (用 --torch cuXXX 可重装) =="
    else
      echo "!! 未检测到 torch。请用 --torch cu128|cu121|... 指定 CUDA 索引安装," >&2
      echo "   或先手动装好与 GPU 匹配的 torch, 再 --pip。" >&2
      echo "   (RTX 5090 等 Blackwell 需 cu128)" >&2
    fi
  fi
  echo "== pip install -r env/requirements.txt =="
  pip install -r "${CODE_ROOT}/env/requirements.txt"
}

step_models() {
  dl_hf "$VANILLA_REPO" "model" "${MODEL_DIR}/${VANILLA_NAME}"   # 角色: vanilla
  dl_hf "$ORIGIN_REPO"  "model" "${MODEL_DIR}/${ORIGIN_NAME}"    # 角色: origin(=参考项目 oracle)
}

step_data() {
  dl_hf "$DATA_REPO" "dataset" "${DATA_DIR}/${DATA_NAME}"
}

step_vllm() {
  local spec="vllm"
  [[ -n "$VLLM_VERSION" ]] && spec="vllm==${VLLM_VERSION}"
  echo "== pip install ${spec} =="
  pip install "$spec"
  echo "== 验证 vllm import =="
  python -c "import vllm; print('vllm', vllm.__version__)"
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pip)      DO_PIP=1; SPECIFIED=1; shift ;;
    --torch)    map_cuda_to_index "${2:-auto}"; DO_PIP=1; SPECIFIED=1; shift ${2:+2} || shift ;;
    --models)   DO_MODELS=1; SPECIFIED=1; shift ;;
    --data)     DO_DATA=1; SPECIFIED=1; shift ;;
    --vllm)     DO_VLLM=1; SPECIFIED=1
                shift
                if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then VLLM_VERSION="$1"; shift; fi ;;
    --all)      DO_PIP=1; DO_MODELS=1; DO_DATA=1; DO_VLLM=1; SPECIFIED=1; shift ;;
    --force)    FORCE=1; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --yes)      YES=1; shift ;;
    -h|--help)  usage ;;
    *) echo "!! 未知参数: $1" >&2; usage ;;
  esac
done

if [[ "$SPECIFIED" == 0 ]]; then DO_PIP=1; DO_MODELS=1; DO_DATA=1; fi

detect_gpu

# ---- 打印计划 -------------------------------------------------------------
cat <<EOF

=================== setup_env 执行计划 ===================
  CODE_ROOT : ${CODE_ROOT}
  DEP_ROOT  : ${DEP_ROOT}
  MODEL_DIR : ${MODEL_DIR}
  DATA_DIR  : ${DATA_DIR}
  HF_ENDPOINT: ${HF_ENDPOINT}
  GPU       : ${GPU_NAME}
  CUDA(drv) : ${CUDA_VER}
  ---- 步骤 ----
  pip      : $([[ $DO_PIP = 1 ]] && echo yes || echo no)    (torch-index: ${TORCH_INDEX:-默认跳过torch})
  models   : $([[ $DO_MODELS = 1 ]] && echo yes || echo no)
              vanilla -> ${VANILLA_REPO}  (${MODEL_DIR}/${VANILLA_NAME})
              origin  -> ${ORIGIN_REPO}   (${MODEL_DIR}/${ORIGIN_NAME})   [参考项目称 oracle]
  data     : $([[ $DO_DATA = 1 ]] && echo yes || echo no)
              ${DATA_REPO}  -> ${DATA_DIR}/${DATA_NAME}
  vllm     : $([[ $DO_VLLM = 1 ]] && echo yes || echo no)    (${VLLM_VERSION:-latest})
  force    : ${FORCE}
===========================================================

EOF

if [[ "$DRY_RUN" == 1 ]]; then
  echo "[dry-run] 计划如上, 未执行任何操作。"
  exit 0
fi
if [[ "$YES" != 1 ]]; then
  if [[ -t 0 ]]; then
    read -r -p "确认执行? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "已取消。"; exit 1; }
  else
    echo "!! 非交互终端且未给 --yes, 默认继续; 如需预览用 --dry-run。" >&2
  fi
fi

# 日志: 控制台 + 落盘(便于复查)
LOG_FILE="${SCRIPT_DIR}/setup_env_${TS}.log"
exec > >(tee "$LOG_FILE") 2>&1
echo "== 日志: ${LOG_FILE} =="

if [[ "$DO_PIP" == 1 ]]; then    step_pip; fi
if [[ "$DO_MODELS" == 1 ]]; then step_models; fi
if [[ "$DO_DATA" == 1 ]]; then   step_data; fi
if [[ "$DO_VLLM" == 1 ]]; then   step_vllm; fi

cat <<EOF

=================== setup_env 完成 ===================
  模型: ${MODEL_DIR}/{${VANILLA_NAME}, ${ORIGIN_NAME}}
  数据: ${DATA_DIR}/${DATA_NAME}
  日志: ${LOG_FILE}
=======================================================
注意: 实验代码目前仍按 repo_id + local_files_only(命中 HF 缓存)加载。
让代码改从 ${DEP_ROOT} 本地路径加载(角色 vanilla/origin)属于 Phase B 接线工作,
本脚本只负责下载落盘, 下载完成后勿直接跑实验。
EOF
