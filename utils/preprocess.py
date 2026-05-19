"""根据 configs/preprocess.py 配置 Deimv2 AutoImageProcessor 与 letterbox 几何变换。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

from configs import preprocess as pc
from utils.common import as_pretrained_identifier


def uses_letterbox() -> bool:
    return pc.uses_letterbox()


def preprocess_settings_dict() -> dict[str, Any]:
    """当前 preprocess 配置快照（写入 meta.json 等，供自研 C++ 部署读取）。"""
    return {
        "input_size": pc.INPUT_SIZE,
        "resize_mode": pc.RESIZE_MODE,
        "letterbox_fill_rgb": list(pc.LETTERBOX_FILL_RGB),
        "do_resize": pc.DO_RESIZE,
        "do_rescale": pc.DO_RESCALE,
        "do_normalize": pc.DO_NORMALIZE,
        "image_mean": list(pc.IMAGE_MEAN),
        "image_std": list(pc.IMAGE_STD),
        "use_checkpoint_preprocessor": pc.USE_CHECKPOINT_PREPROCESSOR,
        "force_apply_config": pc.FORCE_APPLY_CONFIG,
    }


def letterbox_params(
    orig_w: int, orig_h: int, input_size: int | None = None
) -> tuple[float, int, int, int, int]:
    """返回 (ratio, new_w, new_h, pad_w, pad_h)。"""
    input_size = int(input_size or pc.INPUT_SIZE)
    ratio = min(float(input_size) / max(orig_w, 1), float(input_size) / max(orig_h, 1))
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2
    return ratio, new_w, new_h, pad_w, pad_h


def letterbox_pil_coco(
    image: Image.Image,
    annotations: list[dict],
    *,
    input_size: int | None = None,
    fill_rgb: tuple[int, int, int] | None = None,
) -> tuple[Image.Image, list[dict]]:
    """PIL + COCO xywh 标注：letterbox 到 input_size 正方形。"""
    input_size = int(input_size or pc.INPUT_SIZE)
    fill_rgb = fill_rgb if fill_rgb is not None else pc.LETTERBOX_FILL_RGB
    w, h = image.size
    if w <= 0 or h <= 0:
        return image, annotations

    ratio, new_w, new_h, pad_w, pad_h = letterbox_params(w, h, input_size)
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (input_size, input_size), fill_rgb)
    canvas.paste(resized, (pad_w, pad_h))

    out_anns: list[dict] = []
    for ann in annotations:
        if int(ann.get("iscrowd", 0)) == 1:
            continue
        bbox = ann.get("bbox")
        if bbox is None:
            continue
        x, y, bw, bh = (float(v) for v in bbox)
        if bw <= 0 or bh <= 0:
            continue
        x1 = x * ratio + pad_w
        y1 = y * ratio + pad_h
        x2 = (x + bw) * ratio + pad_w
        y2 = (y + bh) * ratio + pad_h
        x1 = max(0.0, min(x1, float(input_size - 1)))
        y1 = max(0.0, min(y1, float(input_size - 1)))
        x2 = max(0.0, min(x2, float(input_size)))
        y2 = max(0.0, min(y2, float(input_size)))
        bw2, bh2 = x2 - x1, y2 - y1
        if bw2 <= 1.0 or bh2 <= 1.0:
            continue
        out_anns.append(
            {
                **{k: v for k, v in ann.items() if k != "bbox"},
                "bbox": [x1, y1, bw2, bh2],
                "area": float(bw2 * bh2),
            }
        )
    return canvas, out_anns


def letterbox_inverse_xyxy(
    box: np.ndarray | list[float],
    *,
    ratio: float,
    pad_w: int,
    pad_h: int,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """letterbox 画布上的 xyxy 映回原图像素坐标（自研 C++ 后处理可参考）。"""
    x1, y1, x2, y2 = (float(v) for v in box)
    inv = 1.0 / ratio if ratio > 1e-9 else 1.0
    x1 = (x1 - pad_w) * inv
    y1 = (y1 - pad_h) * inv
    x2 = (x2 - pad_w) * inv
    y2 = (y2 - pad_h) * inv
    x1 = max(0.0, min(x1, float(orig_w - 1)))
    y1 = max(0.0, min(y1, float(orig_h - 1)))
    x2 = max(0.0, min(x2, float(orig_w)))
    y2 = max(0.0, min(y2, float(orig_h)))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def apply_preprocess_config(processor: AutoImageProcessor) -> AutoImageProcessor:
    processor.do_rescale = bool(pc.DO_RESCALE)
    processor.do_normalize = bool(pc.DO_NORMALIZE)
    if pc.DO_NORMALIZE:
        processor.image_mean = list(pc.IMAGE_MEAN)
        processor.image_std = list(pc.IMAGE_STD)
    if uses_letterbox():
        processor.do_resize = False
        if hasattr(processor, "do_pad"):
            processor.do_pad = False
    else:
        processor.do_resize = bool(pc.DO_RESIZE)
        if pc.DO_RESIZE and hasattr(processor, "size"):
            processor.size = {"height": int(pc.INPUT_SIZE), "width": int(pc.INPUT_SIZE)}
    return processor


def _is_local_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (path / "preprocessor_config.json").is_file()


def load_deimv2_processor(model_id_or_path: str | Path) -> AutoImageProcessor:
    load_target = as_pretrained_identifier(model_id_or_path)
    processor = AutoImageProcessor.from_pretrained(load_target)

    local_dir = Path(model_id_or_path) if isinstance(model_id_or_path, Path) else Path(load_target)
    use_saved = _is_local_checkpoint_dir(local_dir) and pc.USE_CHECKPOINT_PREPROCESSOR
    if use_saved and not pc.FORCE_APPLY_CONFIG:
        return processor

    return apply_preprocess_config(processor)


def describe_processor(processor: AutoImageProcessor) -> str:
    if uses_letterbox():
        geom = f"Letterbox→{pc.INPUT_SIZE}×{pc.INPUT_SIZE} fill={pc.LETTERBOX_FILL_RGB}"
    else:
        size = getattr(processor, "size", None)
        h = getattr(size, "height", "?") if size else "?"
        w = getattr(size, "width", "?") if size else "?"
        geom = f"Resize→{h}×{w}"
    parts = [f"resize_mode={pc.RESIZE_MODE}", geom]
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
        "--resize-mode",
        choices=("stretch", "letterbox"),
        default=None,
        help=f"几何缩放：stretch=拉伸；letterbox=等比+黑边，默认 {pc.RESIZE_MODE}",
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
    if getattr(args, "resize_mode", None) is not None:
        pc.RESIZE_MODE = args.resize_mode  # type: ignore[assignment]
        applied.append(f"resize_mode={pc.RESIZE_MODE}")
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