"""
将 HuggingFace 格式的 Deimv2 checkpoint 导出为 ONNX。

导出 ONNX 仅含模型前向（logits + pred_boxes）；RT-DETR 风格 top-k 后处理在 Python 中完成
（utils.postprocess.postprocess_detections），避免 top-k / SDPA 编入 ONNX 后与 onnxruntime 不一致。
导出前会将 attention 设为 eager；trace/export 与验证固定使用 CPU。置信度阈值与 NMS 请在推理端完成。

用法：
  python export_onnx.py outputs/deimv2_s_march/checkpoint-best
  # 默认输出 onnx_models/deimv2_s_march_epoch<N>.onnx（从 checkpoint 解析 epoch，不用 checkpoint-best）
  python export_onnx.py -o onnx_models/mymodel_epoch50.onnx --check --verify

配置：configs/export_onnx.py（CHECKPOINT、OPSET、VERIFY 等默认值）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import Deimv2ForObjectDetection
from transformers.utils import logging as hf_logging

import configs.export_onnx as cfg
from utils.onnx import (
    Deimv2OnnxCore,
    default_output_path,
    export_onnx,
    onnx_check,
    onnx_simplify,
    prepare_model_for_onnx_export,
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
    p.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=cfg.CHECKPOINT,
        help="HF checkpoint 目录（含 config.json）；默认 configs/export_onnx.CHECKPOINT",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=cfg.OUTPUT,
        help="输出 .onnx 路径；默认 onnx_models/<run>_epochN.onnx（从 checkpoint 解析 epoch）",
    )
    p.add_argument("--opset", type=int, default=cfg.OPSET, help="ONNX opset 版本")
    p.add_argument(
        "--input-size",
        type=int,
        default=cfg.INPUT_SIZE,
        help="导出时 dummy 输入边长（须与训练 processor size 一致）",
    )
    p.add_argument(
        "--dynamic-batch",
        action=argparse.BooleanOptionalAction,
        default=cfg.DYNAMIC_BATCH,
        help="启用 batch 维 dynamic_axes（默认仅固定 batch=1）",
    )
    p.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=cfg.VERIFY,
        help="导出后用 onnxruntime 对比 PyTorch 与 ONNX 输出（需安装 onnxruntime）",
    )
    p.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=cfg.CHECK,
        help="导出后用 onnx.checker.check_model 校验（需 pip install onnx）",
    )
    p.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=cfg.SIMPLIFY,
        help="导出后用 onnxsim 简化图并覆盖原文件（需 pip install onnxsim）",
    )
    add_preprocess_arguments(p)
    args = p.parse_args(argv)
    if args.checkpoint is None:
        p.error("请提供 checkpoint 路径，或在 configs/export_onnx.py 中设置 CHECKPOINT")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prep_applied = apply_preprocess_from_namespace(args)
    if prep_applied:
        print("预处理 CLI 覆盖: " + ", ".join(prep_applied))
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    onnx_path = (args.output or default_output_path(checkpoint_dir)).resolve()

    print(f"checkpoint: {checkpoint_dir}")
    print(f"output:     {onnx_path}")

    processor = load_deimv2_processor(checkpoint_dir)
    model = prepare_model_for_onnx_export(
        Deimv2ForObjectDetection.from_pretrained(str(checkpoint_dir))
    )

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
