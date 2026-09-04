"""依赖路径助手：实验代码从 dependencies/{models,data} 本地加载。

所有"默认模型/数据路径"都应经由此模块解析，便于在服务器上统一指向
一次性下载好的 dependencies 目录（可用环境变量覆盖，方便调试/换机）。
"""
import os

_CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/
_DEP_ROOT = os.environ.get("DEP_ROOT") or os.path.join(os.path.dirname(_CODE_ROOT), "dependencies")
MODEL_DIR = os.environ.get("DEP_MODELS") or os.path.join(_DEP_ROOT, "models")
DATA_DIR = os.environ.get("DEP_DATA") or os.path.join(_DEP_ROOT, "data")


def model_dir(name: str) -> str:
    return os.path.join(MODEL_DIR, name)


def umu_bench_dir() -> str:
    return os.environ.get("DATA_SPLIT_DIR") or os.path.join(DATA_DIR, "UMU-bench")


# 角色模型（术语: 参考项目称 oracle 的微调基线，本项目称 origin）
VANILLA = model_dir("llava-1.5-7b-hf")  # 原始基座（直接评估 / retrain 基座）
ORIGIN = model_dir("llava_smu_ft")      # 微调基线（遗忘方法的 policy/π_ref 起点）
UMU_BENCH = umu_bench_dir()             # 数据集根（含 forget_5/retain_95/real_person/full_data）


def is_llava(ref) -> bool:
    """宽容识别 LLaVA 模型族：接受 repo id(如 llava-hf/llava-1.5-7b-hf)
    或本地路径(目录名以 llava 开头)，替代脆弱的 model_id.startswith("llava")。"""
    s = str(ref).rstrip("/")
    base = os.path.basename(s)
    return base.lower().startswith("llava") or s.lower().startswith("llava")
