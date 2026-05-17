"""根据 configs/preprocess.py 配置 Deimv2 AutoImageProcessor。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoImageProcessor

from configs import preprocess as pc


def preprocess_settings_dict() -> dict[str, Any]:
    """当前 preprocess 配置快照（写入 meta.json 等）。"""
    return {
        "input_size": pc.INPUT_SIZE,
        "do_resize": pc.DO_RESIZE,
        "do_rescale": pc.DO_RESCALE,
        "do_normalize": pc.DO_NORMALIZE,
        "image_mean": list(pc.IMAGE_MEAN),
        "image_std": list(pc.IMAGE_STD),
        "use_checkpoint_preprocessor": pc.USE_CHECKPOINT_PREPROCESSOR,
        "force_apply_config": pc.FORCE_APPLY_CONFIG,
    }


def apply_preprocess_config(processor: AutoImageProcessor) -> AutoImageProcessor:
    processor.do_resize = bool(pc.DO_RESIZE)
    processor.do_rescale = bool(pc.DO_RESCALE)
    processor.do_normalize = bool(pc.DO_NORMALIZE)
    if pc.DO_NORMALIZE:
        processor.image_mean = list(pc.IMAGE_MEAN)
        processor.image_std = list(pc.IMAGE_STD)
    if pc.DO_RESIZE and hasattr(processor, "size"):
        processor.size = {"height": int(pc.INPUT_SIZE), "width": int(pc.INPUT_SIZE)}
    return processor


def _is_local_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (path / "preprocessor_config.json").is_file()


def load_deimv2_processor(model_id_or_path: str | Path) -> AutoImageProcessor:
    path = Path(model_id_or_path)
    processor = AutoImageProcessor.from_pretrained(str(path))

    use_saved = _is_local_checkpoint_dir(path) and pc.USE_CHECKPOINT_PREPROCESSOR
    if use_saved and not pc.FORCE_APPLY_CONFIG:
        return processor

    return apply_preprocess_config(processor)


def describe_processor(processor: AutoImageProcessor) -> str:
    size = getattr(processor, "size", None)
    h = getattr(size, "height", "?") if size else "?"
    w = getattr(size, "width", "?") if size else "?"
    parts = [f"Resize→{h}×{w}"]
    if getattr(processor, "do_rescale", True):
        parts.append("÷255")
    if getattr(processor, "do_normalize", False):
        parts.append(
            f"normalize mean={list(processor.image_mean)} std={list(processor.image_std)}"
        )
    else:
        parts.append("无 mean/std 归一化")
    return " → ".join(parts)


# ---------------------------------------------------------------------------
# CLI 覆盖 configs/preprocess.py
# ---------------------------------------------------------------------------

import argparse


def add_preprocess_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("输入预处理（覆盖 configs/preprocess.py）")
    g.add_argument(
        "--input_size",
        type=int,
        default=None,
        help=f"网络输入边长，默认 {pc.INPUT_SIZE}",
    )
    g.add_argument(
        "--do_rescale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否像素÷255",
    )
    g.add_argument(
        "--do_normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否 (x-mean)/std",
    )
    g.add_argument(
        "--image_mean",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=None,
        help="归一化 mean，三通道",
    )
    g.add_argument(
        "--image_std",
        type=float,
        nargs=3,
        metavar=("R", "G", "B"),
        default=None,
        help="归一化 std，三通道",
    )
    g.add_argument(
        "--use_checkpoint_preprocessor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="加载本地 checkpoint 时是否使用其 preprocessor_config.json",
    )
    g.add_argument(
        "--force_preprocess_config",
        action="store_true",
        help="即使加载 checkpoint 也强制套用 configs/preprocess.py（及本次 CLI 覆盖）",
    )


def apply_preprocess_from_namespace(args: argparse.Namespace) -> list[str]:
    applied: list[str] = []

    if getattr(args, "input_size", None) is not None:
        pc.INPUT_SIZE = int(args.input_size)
        applied.append(f"input_size={pc.INPUT_SIZE}")
    if getattr(args, "do_rescale", None) is not None:
        pc.DO_RESCALE = bool(args.do_rescale)
        applied.append(f"do_rescale={pc.DO_RESCALE}")
    if getattr(args, "do_normalize", None) is not None:
        pc.DO_NORMALIZE = bool(args.do_normalize)
        applied.append(f"do_normalize={pc.DO_NORMALIZE}")
    if getattr(args, "image_mean", None) is not None:
        pc.IMAGE_MEAN = tuple(float(x) for x in args.image_mean)
        applied.append(f"image_mean={list(pc.IMAGE_MEAN)}")
    if getattr(args, "image_std", None) is not None:
        pc.IMAGE_STD = tuple(float(x) for x in args.image_std)
        applied.append(f"image_std={list(pc.IMAGE_STD)}")
    if getattr(args, "use_checkpoint_preprocessor", None) is not None:
        pc.USE_CHECKPOINT_PREPROCESSOR = bool(args.use_checkpoint_preprocessor)
        applied.append(f"use_checkpoint_preprocessor={pc.USE_CHECKPOINT_PREPROCESSOR}")
    if getattr(args, "force_preprocess_config", False):
        pc.FORCE_APPLY_CONFIG = True
        applied.append("force_preprocess_config=True")

    return applied