"""
从 outputs 下各 checkpoint-epoch*/train_metrics.json 读取指标并绘制曲线。
用法:
  python plot_training_curves.py
  python plot_training_curves.py --run_dir E:\\teaDetector\\outputs\\deimv2_s_tealeaves --epochs 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.setdefault("font.sans-serif", ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"])
plt.rcParams["axes.unicode_minus"] = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir",
        type=Path,
        default=Path(r"E:\teaDetector\outputs\deimv2_s_tealeaves"),
    )
    p.add_argument("--epochs", type=int, default=50, help="绘制前 N 个 epoch（从 1 起）")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 PNG 路径，默认 run_dir/training_curves_epoch{N}.png",
    )
    return p.parse_args()


def load_series(run_dir: Path, max_epoch: int) -> tuple[list[int], list[float], list[float], list[float]]:
    epochs: list[int] = []
    train_loss: list[float] = []
    train_map: list[float] = []
    val_map: list[float] = []
    for e in range(1, max_epoch + 1):
        p = run_dir / f"checkpoint-epoch{e}" / "train_metrics.json"
        if not p.is_file():
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        epochs.append(int(d["epoch"]))
        train_loss.append(float(d["train_loss"]))
        tm = d.get("train_map") or {}
        vm = d.get("val_map") or {}
        # 兼容旧格式顶层 bbox_mAP
        train_map.append(float(tm.get("bbox_mAP", d.get("bbox_mAP", 0.0))))
        val_map.append(float(vm.get("bbox_mAP", 0.0)))
    return epochs, train_loss, train_map, val_map


def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    epochs, train_loss, train_map, val_map = load_series(run_dir, args.epochs)
    if not epochs:
        raise SystemExit(f"未在 {run_dir} 找到 checkpoint-epoch1..{args.epochs} 的 train_metrics.json")

    out_path = args.out or (run_dir / f"training_curves_epoch{epochs[-1]}.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(epochs, train_loss, color="#1f77b4", linewidth=1.8, marker="o", markersize=3)
    axes[0].set_ylabel("train_loss")
    axes[0].set_title("Train loss & COCO bbox mAP (train / val)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_map, color="#2ca02c", linewidth=1.8, marker="o", markersize=3, label="train bbox_mAP")
    axes[1].plot(epochs, val_map, color="#d62728", linewidth=1.8, marker="s", markersize=3, label="val bbox_mAP")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mAP @ IoU=0.50:0.95")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
