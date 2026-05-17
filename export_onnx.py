"""
将 HuggingFace 格式的 Deimv2 checkpoint 导出为 ONNX。

导出 ONNX 仅含模型前向（logits + pred_boxes）；RT-DETR 风格 top-k 后处理在 Python 中完成
（utils.postprocess.postprocess_detections），避免 top-k / SDPA 编入 ONNX 后与 onnxruntime 不一致。
导出前会将 attention 设为 eager。置信度阈值与 NMS 请在推理端完成。

用法：
  python export_onnx.py outputs/deimv2_s_march/checkpoint-best
  python export_onnx.py -r outputs/deimv2_s_march/final -o onnx_models/mymodel.onnx --check --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import Deimv2ForObjectDetection
from transformers.utils import logging as hf_logging

import configs.eval as _eval_cfg
from utils.common import resolve_torch_device
from utils.onnx import (
    Deimv2OnnxCore,
    default_output_path,
    export_onnx,
    onnx_check,
    onnx_simplify,
    resolve_checkpoint_dir,
    save_export_meta,
    verify_export,
)
from utils.postprocess import postprocess_config_from_model
from utils.preprocess import (
    add_preprocess_arguments,
    apply_preprocess_from_namespace,
    describe_processor,
    load_deimv2_processor,
)

hf_logging.disable_progress_bar()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deimv2 checkpoint → ONNX")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=None,
        help="HF checkpoint 目录（含 config.json 等），与 -r 二选一",
    )
    src.add_argument(
        "-r",
        "--resume",
        type=Path,
        default=None,
        help="同官方 DEIMv2：checkpoint 目录路径",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 .onnx 路径；默认 onnx_models/<run>_<tag>.onnx",
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="导出设备 cuda / cpu；默认自动",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=_eval_cfg.INPUT_SIZE,
        help="导出时 dummy 输入边长（须与训练 processor size 一致）",
    )
    p.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="启用 batch 维 dynamic_axes（默认仅固定 batch=1）",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="导出后用 onnxruntime 对比 PyTorch 与 ONNX 输出（需安装 onnxruntime）",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="导出后用 onnx.checker.check_model 校验（需 pip install onnx）",
    )
    p.add_argument(
        "--simplify",
        action="store_true",
        help="导出后用 onnxsim 简化图并覆盖原文件（需 pip install onnxsim）",
    )
    add_preprocess_arguments(p)
    args = p.parse_args(argv)
    ckpt = args.checkpoint if args.checkpoint is not None else args.resume
    if ckpt is None:
        p.error("请提供 checkpoint 路径（位置参数）或 -r/--resume")
    args.checkpoint = ckpt
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prep_applied = apply_preprocess_from_namespace(args)
    if prep_applied:
        print("预处理 CLI 覆盖: " + ", ".join(prep_applied))
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    onnx_path = (args.output or default_output_path(checkpoint_dir)).resolve()
    device = resolve_torch_device(args.device)

    print(f"checkpoint: {checkpoint_dir}")
    print(f"output:     {onnx_path}")
    print(f"device:     {device}")

    processor = load_deimv2_processor(checkpoint_dir)
    model = Deimv2ForObjectDetection.from_pretrained(str(checkpoint_dir))
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    model.eval().cpu()
    if device.type != "cpu":
        print(f"提示: 已忽略 --device {device}，导出与验证固定使用 CPU（保证 ONNX 数值一致）")

    proc_h = getattr(processor.size, "height", None) or args.input_size
    proc_w = getattr(processor.size, "width", None) or args.input_size
    if proc_h != args.input_size or proc_w != args.input_size:
        print(
            f"提示: processor size={proc_h}x{proc_w}，--input-size={args.input_size}；"
            "推理时请与 processor 配置一致。"
        )
    print(f"预处理（与训练 checkpoint 一致）: {describe_processor(processor)}")

    post_cfg = postprocess_config_from_model(model)
    core = Deimv2OnnxCore(model)
    export_onnx(
        core,
        onnx_path,
        input_size=args.input_size,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        export_device=torch.device("cpu"),
    )
    print(f"已写入: {onnx_path}")

    meta_path = onnx_path.with_suffix(".meta.json")
    save_export_meta(
        meta_path,
        checkpoint_dir=checkpoint_dir,
        onnx_path=onnx_path,
        input_size=args.input_size,
        num_queries=int(post_cfg["num_queries"]),
        num_classes=int(post_cfg["num_classes"]),
        use_focal_loss=bool(post_cfg["use_focal_loss"]),
        opset=args.opset,
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
    )
    print(f"元数据:   {meta_path}")

    if args.check:
        onnx_check(onnx_path)
    if args.simplify:
        onnx_simplify(onnx_path, input_size=args.input_size)
    if args.verify:
        print("验证 ONNX …")
        verify_export(model, processor, onnx_path, post_cfg)


if __name__ == "__main__":
    main()
