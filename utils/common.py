"""设备、项目路径、模型路径命名、HF 预设、matplotlib 中文字体。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def dataset_path(name: str) -> Path:
    """数据集根目录：`datasets/<name>`。"""
    return PROJECT_ROOT / "datasets" / name


Backend = Literal["onnx", "checkpoint"]

_FONT_CANDIDATE_FILES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/System/Library/Fonts/STHeiti Light.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
]

HF_DEIMV2_PRESETS: dict[str, str] = {
    "dinov3_s": "harshaljanjani/DEIMv2_DINOv3_S_COCO_Transformers",
    "dinov3_l": "alessioarcara/deimv2-deimv2_dinov3_l_coco",
}


def resolve_torch_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_backend(model_path: Path) -> Backend:
    if model_path.suffix.lower() == ".onnx":
        return "onnx"
    return "checkpoint"


def make_safe_name(text: str) -> str:
    chars = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def resolve_checkpoint_epoch(model_path: Path) -> int | None:
    """从目录名 checkpoint-epochN 或 train_metrics.json / training_state.pt 读取 epoch。"""
    name = model_path.name
    prefix = "checkpoint-epoch"
    if name.lower().startswith(prefix):
        try:
            return int(name[len(prefix) :])
        except ValueError:
            pass
    metrics_path = model_path / "train_metrics.json"
    if metrics_path.is_file():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                return int(json.load(f)["epoch"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    state_path = model_path / "training_state.pt"
    if state_path.is_file():
        try:
            import torch

            state = torch.load(state_path, map_location="cpu", weights_only=False)
            ep = state.get("epoch")
            if ep is not None:
                return int(ep)
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
    return None


def checkpoint_onnx_basename(checkpoint_dir: Path) -> str:
    """ONNX 默认文件名（无扩展名），如 deimv2_l_march_epoch143（不用 checkpoint-best）。"""
    parent = make_safe_name(checkpoint_dir.parent.name)
    leaf = checkpoint_dir.name
    if leaf in ("checkpoint-best", "final") or leaf.startswith("checkpoint-"):
        ep = resolve_checkpoint_epoch(checkpoint_dir)
        if ep is not None:
            return f"{parent}_epoch{ep}"
        if leaf in ("checkpoint-best", "final"):
            return parent
    return f"{parent}_{make_safe_name(leaf)}"


def display_model_name(model_path: Path, backend: Backend) -> str:
    """图表/汇总用的短名。HF 权重目录用「父目录-epochN」，避免多个 checkpoint-best 重名。"""
    if backend == "onnx":
        return make_safe_name(model_path.stem)
    leaf = model_path.name
    parent = model_path.parent.name
    if leaf in ("checkpoint-best", "final") or leaf.startswith("checkpoint-"):
        ep = resolve_checkpoint_epoch(model_path)
        if ep is not None:
            return make_safe_name(f"{parent}-epoch{ep}")
        return make_safe_name(parent)
    return make_safe_name(leaf)


def resolve_pretrained_hub_id(pretrained: str) -> str:
    """返回 HuggingFace Hub repo id。"""
    if pretrained not in HF_DEIMV2_PRESETS:
        raise ValueError(f"未知 --pretrained={pretrained!r}，可选: {tuple(HF_DEIMV2_PRESETS)}")
    return HF_DEIMV2_PRESETS[pretrained]


def as_pretrained_identifier(path_or_id: str | Path) -> str:
    """传给 transformers.from_pretrained 的标识：本地路径 resolve；Hub repo id 保留 '/'。

    Windows 上 Path('org/model') 会变成 org\\model，导致 HuggingFace 校验失败。
    """
    if isinstance(path_or_id, Path):
        return str(path_or_id.resolve())
    p = Path(path_or_id)
    if p.exists():
        return str(p.resolve())
    if "/" in path_or_id and "\\" not in path_or_id:
        if len(path_or_id) < 2 or path_or_id[1] != ":":
            return path_or_id
    return str(p.resolve() if p.is_absolute() else p)


def setup_chinese_font() -> str:
    """注册并启用系统中文字体；返回字体名，未找到则返回空字符串。"""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    for fp in _FONT_CANDIDATE_FILES:
        if not fp.is_file():
            continue
        try:
            fm.fontManager.addfont(str(fp))
            name = fm.FontProperties(fname=str(fp)).get_name()
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
        except (OSError, RuntimeError):
            continue

    for family in (
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ):
        try:
            path = fm.findfont(family, fallback_to_default=False)
            if path and "DejaVu" not in Path(path).name:
                plt.rcParams["font.family"] = "sans-serif"
                plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return family
        except (ValueError, OSError):
            continue

    plt.rcParams["axes.unicode_minus"] = False
    return ""


def setup_matplotlib_chinese():
    """Agg 后端 + 中文字体，返回 pyplot 模块。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup_chinese_font()
    return plt
