#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — 新服务器一键环境配置（单入口：环境 + 依赖 + 模型/数据集 + 可选 vLLM/验证）
# =============================================================================
# 说明:
#   1) 模型按角色落到 dependencies/models/<name>/, 数据集落到 dependencies/data/UMU-bench/。
#   2) 术语: 参考项目称 "oracle" 的微调基线, 本项目统一称 "origin"。
#        vanilla = llava-hf/llava-1.5-7b-hf ;  origin = llava_smu_ft
#   3) xet/软链禁用 + 镜像下载; 幂等(已完成则跳过, --force 重下)。
#   4) 首次运行会自动创建 conda 环境(默认 maw, python 3.10), 之后全部步骤在该 env 内执行。
#   5) 下载模型/数据集/装库分别委托更合适的工具(pip / huggingface_hub 的 hf CLI / conda)。
#   6) 服务器上必须用 tmux 运行(长时间下载防断连)。
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 配置区 —— 均可用环境变量覆盖
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="${CODE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DEP_ROOT="${DEP_ROOT:-${CODE_ROOT}/../dependencies}"
MODEL_DIR="${MODEL_DIR:-${DEP_ROOT}/models}"
DATA_DIR="${DATA_DIR:-${DEP_ROOT}/data}"

ENV_NAME="${ENV_NAME:-maw}"          # conda 环境名
ENV_PY="${ENV_PY:-3.10}"             # python 版本
PIP_INDEX="${PIP_INDEX:-}"           # pip 镜像(预设或完整 URL), 空 = 不动 pip 配置
CONDA_CHANNEL="${CONDA_CHANNEL:-defaults}"   # conda channel(镜像如 https://mirrors.ustc.edu.cn/anaconda/pkgs/main)

# HF 镜像与传输开关
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_ENDPOINT
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_SYMLINKS="${HF_HUB_DISABLE_SYMLINKS:-1}"

# 模型/数据集源前缀: chengyewang = 镜像(蓝本验证过); linbojunzi = 上游官方
HF_SOURCE="${HF_SOURCE:-chengyewang}"

VANILLA_REPO="${VANILLA_REPO:-llava-hf/llava-1.5-7b-hf}"
VANILLA_NAME="llava-1.5-7b-hf"
ORIGIN_REPO="${ORIGIN_REPO:-${HF_SOURCE}/llava_smu_ft}"
ORIGIN_NAME="llava_smu_ft"
DATA_REPO="${DATA_REPO:-${HF_SOURCE}/UMU-bench}"
DATA_NAME="UMU-bench"
VLLM_VERSION="${VLLM_VERSION:-}"                            # 空 = 最新稳定版(曾用 0.11.0)

# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
DO_PIP=0 DO_MODELS=0 DO_DATA=0 DO_VLLM=0 DO_VERIFY=0
SPECIFIED=0
FORCE=0 DRY_RUN=0 YES=0
TORCH_INDEX=""
TS="$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<'EOF'
setup_env.sh — 新服务器一键环境配置
  单入口封装: conda 环境 + pip 镜像 + 依赖 + torch + 模型/数据集下载 + (可选)vLLM/验证
  vanilla = llava-hf/llava-1.5-7b-hf ; origin = llava_smu_ft(参考项目称 oracle)
  xet/软链禁用 + 幂等(已完成跳过, --force 重下); 服务器上必须用 tmux 运行

用法:
  ./scripts/setup_env.sh                              # 默认: pip + models + data
  ./scripts/setup_env.sh --all                        # 再加 vLLM
  ./scripts/setup_env.sh --pip-index ustc --yes       # 设 pip 源(预设或URL)后执行默认步骤
  ./scripts/setup_env.sh --pip --torch cu128 --vllm 0.11.0 --models --data --yes
                                                    # 已知CUDA: 一次装好匹配 torch + vllm + 依赖 + 下载
  ./scripts/setup_env.sh --pip --torch cu128 --models --data
  ./scripts/setup_env.sh --verify                     # 只对已下载模型做加载验证(需已装 torch)
  ./scripts/setup_env.sh --dry-run                    # 只打印执行计划

选项:
  --pip            安装依赖 (pip install -r env/requirements.txt)
  --torch [cu]     装匹配 torch/torchvision (cu118|cu121|cu126|cu128|auto; 跳过则检测已装)
  --models         下载模型到 dependencies/models/
  --data           下载数据集到 dependencies/data/UMU-bench/
  --vllm [VER]     安装 vLLM (默认最新, 曾用 0.11.0)
  --verify         下载后/对已有模型做 transformers 加载验证
  --all            等价 --pip --models --data --vllm
  --env-name NAME  conda 环境名(默认 maw), 首次自动创建
  --pip-index SRC  pip 镜像: ustc|aliyun|huawei|tuna|pypi 或完整 URL(写入 ~/.config/pip/pip.conf)
  --force          已存在也重新下载/安装
  --dry-run        只打印执行计划, 不执行
  --yes            跳过确认
EOF
  exit 0
}

# ---------------------------------------------------------------------------
# conda / pip / 预检工具
# ---------------------------------------------------------------------------
detect_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    CUDA_VER="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$' || true)"
  fi
  GPU_NAME="${GPU_NAME:-unknown}"; CUDA_VER="${CUDA_VER:-unknown}"
}

resolve_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then CONDA="${CONDA_EXE}"
  elif command -v conda >/dev/null 2>&1; then CONDA="$(command -v conda)"
  elif [[ -x /root/miniconda3/bin/conda ]]; then CONDA=/root/miniconda3/bin/conda
  elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then CONDA="$HOME/miniconda3/bin/conda"
  else echo "!! 未找到 conda。请先安装 miniconda 或提供 conda 于 PATH。" >&2; exit 1; fi
  local base
  base="$("$CONDA" info --base 2>/dev/null | tr -d '[:space:]')"
  ENV_DIR="${base}/envs/${ENV_NAME}"
}

ensure_env() {
  resolve_conda
  if [[ ! -d "$ENV_DIR" ]]; then
    echo "== 创建 conda 环境 ${ENV_NAME} (python=${ENV_PY}) =="
    # --override-channels: 避免用户自定义 channel(可能不可达/403)导致失败;
    # 国内默认源慢时用 CONDA_CHANNEL 切镜像(如 ustc/tuna 的 anaconda pkgs/main)
    "$CONDA" create -y -n "$ENV_NAME" "python=${ENV_PY}" \
      --override-channels -c "${CONDA_CHANNEL}" -q
  else
    echo "== conda 环境已存在: ${ENV_NAME} =="
  fi
  export PATH="${ENV_DIR}/bin:${PATH}"
  echo "== python: $(python --version) =="
}

write_pip_config() {
  [[ -z "${PIP_INDEX}" ]] && return 0
  local url host
  case "${PIP_INDEX}" in
    ustc)   url="https://mirrors.ustc.edu.cn/pypi/simple/";              host="mirrors.ustc.edu.cn" ;;
    aliyun) url="https://mirrors.aliyun.com/pypi/simple/";               host="mirrors.aliyun.com" ;;
    huawei) url="https://mirrors.huaweicloud.com/repository/pypi/simple/"; host="mirrors.huaweicloud.com" ;;
    tuna)   url="https://pypi.tuna.tsinghua.edu.cn/simple/";             host="pypi.tuna.tsinghua.edu.cn" ;;
    pypi)   url="https://pypi.org/simple/";                              host="pypi.org" ;;
    http*://*|https://*) url="${PIP_INDEX}"; host="$(echo "${PIP_INDEX}" | sed -E 's#^https?://([^/]+)/?.*#\1#')" ;;
    *) echo "!! 未知 pip-index 预设: ${PIP_INDEX} (ustc|aliyun|huawei|tuna|pypi 或完整 URL)" >&2; usage ;;
  esac
  mkdir -p "${HOME}/.config/pip"
  cat > "${HOME}/.config/pip/pip.conf" <<EOF
[global]
index-url = ${url}
trusted-host = ${host}
EOF
  echo "== 已写入 pip 配置: ${HOME}/.config/pip/pip.conf (${url}) =="
}

map_cuda_to_index() {
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

# hf download 封装: 幂等按目录 marker 判断
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
      echo "!! 未检测到 torch。pip -r 将尝试安装默认 torch 构建; 若需匹配 CUDA," >&2
      echo "   请用 --torch cuXXX 显式指定(如 A800 建议 cu126/cu128, RTX5090 需 cu128)。" >&2
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

step_verify() {
  python - <<'PY'
import os, glob, sys, gc
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

MODEL_DIR = os.environ.get("MODEL_DIR", "")
dirs = sorted(
    d for d in glob.glob(os.path.join(MODEL_DIR, "*"))
    if os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json"))
)
if not dirs:
    print("!! MODEL_DIR 下没有可验证的模型目录:", MODEL_DIR)
    sys.exit(1)
ok = True
for d in dirs:
    try:
        print(f"== 加载验证: {d} ==", flush=True)
        m = LlavaForConditionalGeneration.from_pretrained(
            d, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True)
        AutoProcessor.from_pretrained(d, local_files_only=True)
        print(f"OK {os.path.basename(d)} params={m.num_parameters()}", flush=True)
        del m
    except Exception as e:
        ok = False
        print(f"FAIL {d}: {type(e).__name__}: {e}", flush=True)
    gc.collect()
sys.exit(0 if ok else 1)
PY
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pip)         DO_PIP=1; SPECIFIED=1; shift ;;
    --torch)       map_cuda_to_index "${2:-auto}"; DO_PIP=1; SPECIFIED=1; shift ${2:+2} || shift ;;
    --models)      DO_MODELS=1; SPECIFIED=1; shift ;;
    --data)        DO_DATA=1; SPECIFIED=1; shift ;;
    --vllm)        DO_VLLM=1; SPECIFIED=1
                   shift
                   if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then VLLM_VERSION="$1"; shift; fi ;;
    --verify)      DO_VERIFY=1; SPECIFIED=1; shift ;;
    --all)         DO_PIP=1; DO_MODELS=1; DO_DATA=1; DO_VLLM=1; SPECIFIED=1; shift ;;
    --env-name)    ENV_NAME="${2:-maw}"; shift 2 ;;
    --pip-index)   PIP_INDEX="${2:-}"; shift 2 ;;
    --force)       FORCE=1; shift ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --yes)         YES=1; shift ;;
    -h|--help)     usage ;;
    *) echo "!! 未知参数: $1" >&2; usage ;;
  esac
done

if [[ "$SPECIFIED" == 0 ]]; then DO_PIP=1; DO_MODELS=1; DO_DATA=1; fi

detect_gpu
resolve_conda

# ---- 打印计划 -------------------------------------------------------------
cat <<EOF

=================== setup_env 执行计划 ===================
  conda env : ${ENV_NAME} (python=${ENV_PY})  @ ${ENV_DIR}
  CODE_ROOT : ${CODE_ROOT}
  DEP_ROOT  : ${DEP_ROOT}
  MODEL_DIR : ${MODEL_DIR}
  DATA_DIR  : ${DATA_DIR}
  HF_ENDPOINT: ${HF_ENDPOINT}
  pip-index : ${PIP_INDEX:-（不动 pip 配置）}
  GPU       : ${GPU_NAME}   CUDA(drv): ${CUDA_VER}
  ---- 步骤 ----
  pip      : $([[ $DO_PIP = 1 ]] && echo yes || echo no)   (torch-index: ${TORCH_INDEX:-跳过})
  models   : $([[ $DO_MODELS = 1 ]] && echo yes || echo no)
              vanilla -> ${VANILLA_REPO}  (${MODEL_DIR}/${VANILLA_NAME})
              origin  -> ${ORIGIN_REPO}   (${MODEL_DIR}/${ORIGIN_NAME})   [参考项目称 oracle]
  data     : $([[ $DO_DATA = 1 ]] && echo yes || echo no)
              ${DATA_REPO}  -> ${DATA_DIR}/${DATA_NAME}
  vllm     : $([[ $DO_VLLM = 1 ]] && echo yes || echo no)   (${VLLM_VERSION:-latest})
  verify   : $([[ $DO_VERIFY = 1 ]] && echo yes || echo no)
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

ensure_env
write_pip_config
ensure_hf_cli

if [[ "$DO_PIP" == 1 ]]; then    step_pip; fi
if [[ "$DO_MODELS" == 1 ]]; then step_models; fi
if [[ "$DO_DATA" == 1 ]]; then   step_data; fi
if [[ "$DO_VLLM" == 1 ]]; then   step_vllm; fi
if [[ "$DO_VERIFY" == 1 ]]; then step_verify; fi

cat <<EOF

=================== setup_env 完成 ===================
  环境    : ${ENV_NAME} (激活: conda activate ${ENV_NAME})
  模型    : ${MODEL_DIR}/{${VANILLA_NAME}, ${ORIGIN_NAME}}
  数据    : ${DATA_DIR}/${DATA_NAME}
  日志    : ${LOG_FILE}
=======================================================
  实验代码已接线为从 ${DEP_ROOT} 本地路径加载(角色 vanilla/origin)。
  安装与 GPU 匹配的 torch 后, 即可用 run_unlearn.sh / run_eval.sh 跑实验。
EOF
