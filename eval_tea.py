"""
茶叶目标检测统一评测：多模型 × 多数据集（各 train/val）。

指标：AP50、AP75、mAP@[0.50:0.95]（VOC 风格），IoU 列表见 configs/eval.MAP_IOU_THRESHOLDS。
评测结束后在 charts/ 生成模型×数据集对比表（PNG/CSV/Markdown，默认 val 划分）。

用法：
    python eval_tea.py
    python eval_tea.py --model onnx_models/a.onnx outputs/deimv2_s/final
    python eval_tea.py --conf 0.2 --nms 0.3 --val_only
    # mAP 用 MAP_SCORE_THRESHOLD 推理；--conf 仅影响 vis/

仅重绘上次评测图表（不重新推理）：python plot_eval_charts.py --resume outputs/eval/<时间戳目录>

配置：configs/eval.py（DATASETS、MODELS、阈值等）。
可视化抽样由 --seed 固定，不同模型对同一数据集/划分抽取相同图片。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import configs.eval as cfg
from utils.eval import (
    build_run_name,
    build_vis_plan,
    eval_model_on_datasets,
    plot_comparison_charts,
    print_summary_table,
    resolve_dataset_specs,
    validate_results,
)
from utils.preprocess import add_preprocess_arguments, apply_preprocess_from_namespace


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="茶叶检测评测（多模型 × 多数据集）")
    p.add_argument(
        "--model",
        nargs="*",
        type=Path,
        default=None,
        help="一个或多个模型路径；默认 configs/eval.MODELS",
    )
    p.add_argument("--output_dir", type=Path, default=cfg.OUTPUT_DIR)
    p.add_argument("--input_size", type=int, default=cfg.INPUT_SIZE)
    p.add_argument(
        "--conf",
        type=float,
        default=None,
        help=f"仅可视化：绘制预测框的 score 下限；默认 {cfg.CONF_THRESHOLD}",
    )
    p.add_argument(
        "--map_conf",
        type=float,
        default=None,
        help=f"仅 mAP：推理后处理 score 下限（宜低以保留 P-R 候选）；默认 {cfg.MAP_SCORE_THRESHOLD}",
    )
    p.add_argument("--nms", type=float, default=cfg.NMS_THRESHOLD)
    p.add_argument("--vis_num", type=int, default=cfg.VIS_NUM_IMAGES)
    p.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--val_only", action="store_true")
    p.add_argument(
        "--device",
        type=str,
        default=cfg.DEVICE,
        help="cuda / cpu；None 自动。checkpoint 用 torch，ONNX 用对应 ExecutionProvider",
    )
    p.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="仅 ONNX：覆盖 --device，如 CUDAExecutionProvider CPUExecutionProvider",
    )
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=cfg.NUM_WORKERS)
    add_preprocess_arguments(p)
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    prep_applied = apply_preprocess_from_namespace(args)
    if prep_applied:
        print("预处理 CLI 覆盖: " + ", ".join(prep_applied))
    model_paths = list(args.model) if args.model else [Path(p) for p in cfg.MODELS]
    if not model_paths:
        raise SystemExit("未指定 --model，且 configs/eval.MODELS 为空")

    specs = resolve_dataset_specs()
    vis_plan = build_vis_plan(specs, args.vis_num, args.seed, args.val_only)

    run_name = build_run_name(args)
    run_output_dir = args.output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=False)

    print(f"run_name: {run_name}")
    print(f"输出目录: {run_output_dir}")
    print(f"模型数量: {len(model_paths)}，数据集: {[s.name for s in specs]}")
    vis_conf = args.conf if args.conf is not None else cfg.CONF_THRESHOLD
    map_conf = args.map_conf if args.map_conf is not None else cfg.MAP_SCORE_THRESHOLD
    print(f"可视化 seed={args.seed}，每划分 {args.vis_num} 张（全模型共用抽样）")
    print(f"mAP 推理 score≥{map_conf:g}；vis 绘制 score≥{vis_conf:g}")

    models_results = []
    for model_path in model_paths:
        models_results.append(
            eval_model_on_datasets(model_path, specs, vis_plan, run_output_dir, args, vis_conf, map_conf)
        )

    report: dict[str, Any] = {
        "run": {
            "name": run_name,
            "output_dir": str(run_output_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "eval_params": {
            "models": [str(p) for p in model_paths],
            "datasets": [
                {
                    "name": s.name,
                    "image_dir": str(s.image_dir),
                    "train_ann": str(s.train_ann),
                    "val_ann": str(s.val_ann),
                }
                for s in specs
            ],
            "map_score_threshold": map_conf,
            "vis_conf_threshold": vis_conf,
            "conf_threshold": vis_conf,
            "nms_threshold": args.nms,
            "device": args.device,
            "input_size": args.input_size,
            "vis_num": args.vis_num,
            "vis_seed": args.seed,
            "val_only": args.val_only,
            "map_iou_thresholds": list(cfg.MAP_IOU_THRESHOLDS),
        },
        "vis_samples": vis_plan,
        "models": models_results,
    }

    metrics_path = run_output_dir / f"metrics_{run_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n指标已保存: {metrics_path}")

    validate_results(report)
    print_summary_table(report)

    charts_dir = run_output_dir / "charts"
    plot_comparison_charts(report, charts_dir)


if __name__ == "__main__":
    main()
