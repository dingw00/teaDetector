"""
从 outputs 下各 checkpoint-epoch*/train_metrics.json 读取指标并绘制曲线。

每个训练 run 生成一张图（3 行子图）：训练损失、检测 mAP（IoU 0.50:0.95）、检测 mAP@0.5。

默认输出到 outputs/train_curves/（与 outputs/eval 同级）。epoch 数 >100 时不画 marker。

用法:
  python plot_train_curves.py deimv2_l_march deimv2_l_april deimv2_l_march_and_april deimv2_s_march
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils import train as tc


def parse_args():
    p = argparse.ArgumentParser(description="绘制训练 loss / mAP 曲线")
    p.add_argument("run_dirs", nargs="*", type=Path)
    p.add_argument("--run_dir", dest="run_dirs_legacy", nargs="+", type=Path, default=None)
    p.add_argument("--outputs_root", type=Path, default=Path("outputs"))
    p.add_argument("--curves_dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out", type=Path, default=None, help="仅单 run 时有效，仅覆盖 mAP@[.50:.95] 主图路径")
    return p.parse_args()


def main():
    args = parse_args()
    outputs_root = args.outputs_root.expanduser().resolve()
    run_dirs = tc.resolve_run_dirs(args.run_dirs, args.run_dirs_legacy, outputs_root)
    curves_dir = tc.resolve_curves_dir(outputs_root, args.curves_dir)

    if args.out is not None and len(run_dirs) != 1:
        raise SystemExit("--out 仅能在指定单个训练目录时使用")

    if tc._CHINESE_FONT:
        print(f"matplotlib 中文字体: {tc._CHINESE_FONT}", file=sys.stderr)
    else:
        print("警告: 未找到中文字体，图中中文可能显示异常", file=sys.stderr)

    failures: list[str] = []
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            failures.append(f"目录不存在: {run_dir}")
            continue
        try:
            series = tc.load_series(run_dir, args.epochs)
            last = series.epochs[-1]
            out_path = (
                args.out.expanduser().resolve()
                if args.out is not None
                else tc.default_out_path(curves_dir, series.name, last)
            )
            tc.plot_run(series, out_path)
            val_info = ", ".join(series.map_coco.val_per_dataset.keys()) or "—"
            print(f"[{series.name}] epoch {series.epochs[0]}..{last} | val: {val_info}")
            print(f"  已保存: {out_path}")
        except (ValueError, OSError, KeyError) as e:
            failures.append(f"{run_dir}: {e}")

    if failures:
        raise SystemExit("部分目录处理失败:\n  " + "\n  ".join(failures))
    if not run_dirs:
        raise SystemExit("未指定训练目录")


if __name__ == "__main__":
    main()
