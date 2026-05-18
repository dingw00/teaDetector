"""
茶叶目标检测统一评测：多模型 × 多数据集（各 train/val）。

指标：AP50、AP75、mAP@[0.50:0.95]（VOC 风格），IoU 列表见 configs/eval.MAP_IOU_THRESHOLDS。
评测结束后在 charts/ 生成模型×数据集对比表（PNG/CSV/Markdown，默认 val 划分）。

用法：
    python eval_tea.py
    python eval_tea.py --model onnx_models/a.onnx outputs/deimv2_s/final
    python eval_tea.py --conf 0.2 --nms 0.3 --val_only
    python eval_tea.py --vis-conf deimv2_l_march:0.35 dino_0329_30:0.15
    # HF mAP：MAP_SCORE_THRESHOLD + 与 train_tea 相同后处理（无额外 NMS）
    # --nms / NMS_THRESHOLD 仅用于 vis/ 抽样图；--conf / --vis-conf 为 vis 绘制 score 下限
    # 中断后续跑（跳过已完成模型）：
    python eval_tea.py --output_dir outputs/eval/20260517_151507
    python eval_tea.py --resume outputs/eval/20260517_151507
    # 续跑目录下仅重绘 vis/（保留已有 mAP，按当前 vis 阈值）：
    python eval_tea.py --output_dir outputs/eval/20260517_153639 --redraw-vis

仅重绘上次评测图表（不重新推理）：python plot_eval_charts.py --resume outputs/eval/<时间戳目录>

配置：configs/eval.py（DATASETS、MODELS、阈值等）。
可视化抽样由 --seed 固定，不同模型对同一数据集/划分抽取相同图片。
vis/ 下每张抽样图输出一张竖排多模型对比图（subtitle：数据集 | train/val | 图像名 | 模型）。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import configs.eval as cfg
from configs import preprocess as pc
from utils.common import detect_backend, display_model_name
from utils.eval import (
    VisPanelAccumulator,
    build_vis_plan,
    completed_model_paths,
    eval_model_on_datasets,
    normalize_model_path,
    parse_vis_conf_specs,
    plot_comparison_charts,
    print_summary_table,
    redraw_vis_for_model,
    resolve_dataset_specs,
    resolve_eval_run,
    resolve_vis_conf,
    update_model_vis_conf,
    validate_results,
)
from utils.preprocess import add_preprocess_arguments, apply_preprocess_from_namespace


def model_display_stems(model_paths: list[Path]) -> list[str]:
    return [display_model_name(p, detect_backend(p)) for p in model_paths]


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
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="续跑已有评测 run 目录或 metrics_*.json（与 --output_dir 指向 run 目录等效）",
    )
    p.add_argument(
        "--redraw-vis",
        action="store_true",
        help="续跑模式下仅重绘 vis/ 抽样图（保留 metrics 中 mAP，按当前 vis 阈值推理）",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=None,
        help=f"仅可视化：全部模型的默认 score 下限；默认 {cfg.CONF_THRESHOLD}",
    )
    p.add_argument(
        "--vis-conf",
        nargs="*",
        default=None,
        metavar="KEY:THR",
        help=(
            "按模型覆盖可视化阈值（路径/display 名子串，最长匹配）；"
            "例如 deimv2_l_march:0.35；亦可在 configs/eval.VIS_CONF_BY_MODEL 配置"
        ),
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

    run_name, run_output_dir, existing_report, is_resume = resolve_eval_run(args)
    if is_resume:
        run_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_output_dir.mkdir(parents=True, exist_ok=False)

    if existing_report and existing_report.get("vis_samples"):
        vis_plan = existing_report["vis_samples"]
    else:
        vis_plan = build_vis_plan(specs, args.vis_num, args.seed, args.val_only)

    print(f"run_name: {run_name}")
    print(f"输出目录: {run_output_dir}")
    if is_resume:
        n_done = len(existing_report.get("models", [])) if existing_report else 0
        if args.redraw_vis:
            print(f"续跑 + 重绘 vis：已有 {n_done} 个模型 mAP 将保留，全部模型 vis/ 将按当前阈值重画")
        else:
            print(f"续跑模式：已有 {n_done} 个模型结果，将跳过已完成项")
        if existing_report and existing_report.get("vis_samples"):
            print("使用已有 run 的可视化抽样计划")
    elif args.redraw_vis:
        raise SystemExit("--redraw-vis 需指定已有评测 run（--output_dir 或 --resume 指向含 metrics_*.json 的目录）")
    print(f"模型数量: {len(model_paths)}，数据集: {[s.name for s in specs]}")
    vis_conf_default = args.conf if args.conf is not None else cfg.CONF_THRESHOLD
    map_conf = args.map_conf if args.map_conf is not None else cfg.MAP_SCORE_THRESHOLD
    cli_vis_conf = parse_vis_conf_specs(args.vis_conf)
    config_vis_conf = dict(getattr(cfg, "VIS_CONF_BY_MODEL", {}) or {})
    print(f"可视化 seed={args.seed}，每划分 {args.vis_num} 张（全模型共用抽样）")
    print(
        f"HF mAP score≥{map_conf:g}（与 train_tea 一致，无额外 NMS）；"
        f"vis 默认 score≥{vis_conf_default:g}，vis NMS={args.nms:g}"
    )
    if config_vis_conf or cli_vis_conf:
        print(f"vis 按模型覆盖: config={config_vis_conf or '{}'} CLI={cli_vis_conf or '{}'}")

    done_paths = completed_model_paths(existing_report) if existing_report else set()
    models_results = list(existing_report.get("models", [])) if existing_report else []
    vis_accum = VisPanelAccumulator() if vis_plan else None

    if args.redraw_vis:
        for model_path in model_paths:
            backend = detect_backend(model_path)
            vis_conf = resolve_vis_conf(
                model_path,
                backend,
                default=vis_conf_default,
                config_map=config_vis_conf,
                cli_map=cli_vis_conf,
            )
            redraw_vis_for_model(
                model_path, specs, vis_plan, run_output_dir, args, vis_conf, map_conf, vis_accum
            )
            update_model_vis_conf(models_results, model_path, vis_conf)
    else:
        for model_path in model_paths:
            resolved = normalize_model_path(model_path)
            if resolved in done_paths:
                prev = next(
                    (m["name"] for m in models_results if normalize_model_path(m.get("path", "")) == resolved),
                    resolved.name,
                )
                print(f"\n[跳过] 已完成: {model_path} ({prev})")
                continue
            backend = detect_backend(model_path)
            vis_conf = resolve_vis_conf(
                model_path,
                backend,
                default=vis_conf_default,
                config_map=config_vis_conf,
                cli_map=cli_vis_conf,
            )
            models_results.append(
                eval_model_on_datasets(
                    model_path, specs, vis_plan, run_output_dir, args, vis_conf, map_conf, vis_accum
                )
            )

    vis_root = run_output_dir / "vis"
    if vis_accum is not None and model_paths:
        n_vis = vis_accum.compose_all(vis_root, vis_plan, model_display_stems(model_paths))
        if n_vis:
            print(f"多模型对比可视化: {n_vis} 张 → {vis_root}/")

    report: dict[str, Any] = {
        "run": {
            "name": run_name,
            "output_dir": str(run_output_dir),
            "created_at": (existing_report or {}).get("run", {}).get("created_at")
            or datetime.now().isoformat(timespec="seconds"),
            "resumed_at": datetime.now().isoformat(timespec="seconds") if is_resume else None,
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
            "vis_conf_default": vis_conf_default,
            "vis_conf_by_model": {**config_vis_conf, **cli_vis_conf},
            "conf_threshold": vis_conf_default,
            "nms_threshold": args.nms,
            "nms_for_vis_only": True,
            "device": args.device,
            "input_size": pc.INPUT_SIZE,
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
