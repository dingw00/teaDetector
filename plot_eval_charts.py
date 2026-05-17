"""
从上次评测保存的 metrics_*.json 重新生成 charts/（无需重新跑模型推理）。

用法：
    python plot_eval_charts.py --resume outputs/eval/20260516_193045
    python plot_eval_charts.py --resume outputs/eval/run_a outputs/eval/run_b
    python plot_eval_charts.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import configs.eval as cfg
from utils.eval import (
    default_charts_dir,
    load_report,
    regenerate_charts_from_report,
    resolve_metrics_files,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从 metrics_*.json 重绘评测对比图/表")
    p.add_argument(
        "--resume",
        nargs="*",
        type=Path,
        default=None,
        metavar="PATH",
        help="上次评测 run 目录或 metrics_*.json；可写多个。省略则处理 --eval-root 下全部 run",
    )
    p.add_argument(
        "--eval-root",
        type=Path,
        default=cfg.OUTPUT_DIR,
        help=f"未指定 --resume 时扫描的根目录（默认 {cfg.OUTPUT_DIR}）",
    )
    p.add_argument(
        "--charts-dir",
        type=Path,
        default=None,
        help="覆盖 charts 输出目录（仅当 --resume 只对应 1 个 run 时可用）",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.resume is not None and len(args.resume) == 0:
        raise SystemExit("请为 --resume 指定至少一个评测目录或 metrics JSON 路径")

    targets: list[Path] = [] if args.resume is None else list(args.resume)
    metrics_files = resolve_metrics_files(targets, args.eval_root)
    if args.charts_dir is not None and len(metrics_files) != 1:
        raise SystemExit("--charts-dir 仅能在 --resume 只对应 1 个 run 时使用")

    failures: list[str] = []
    for metrics_path in metrics_files:
        try:
            report = load_report(metrics_path)
            charts_dir = default_charts_dir(metrics_path, args.charts_dir)
            print(f"\n=== {metrics_path.parent.name} ===")
            print(f"metrics: {metrics_path}")
            print(f"charts:  {charts_dir}")
            regenerate_charts_from_report(report, charts_dir)
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            failures.append(f"{metrics_path}: {exc}")

    if failures:
        raise SystemExit("部分文件处理失败:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    main()
