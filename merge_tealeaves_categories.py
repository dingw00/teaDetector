"""
将拆分后的 TeaLeaves COCO 数据集类别合并为单类 tea。

默认输入：
datasets/TeaLeavesDatasets_split_lr

默认输出：
datasets/TeaLeavesDatasets_split_lr_tea
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="合并 TeaLeaves COCO 类别为单类 tea")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("datasets") / "TeaLeavesDatasets_split_lr",
        help="输入 COCO 数据集根目录",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("datasets") / "TeaLeavesDatasets_split_lr_tea",
        help="输出 COCO 数据集根目录",
    )
    parser.add_argument("--train_ann", default="annotations/instances_train.json")
    parser.add_argument("--val_ann", default="annotations/instances_val.json")
    parser.add_argument("--overwrite", action="store_true", help="允许删除并重建输出目录")
    return parser.parse_args()


def merge_annotation_file(src_ann: Path, dst_ann: Path):
    with open(src_ann, "r", encoding="utf-8") as f:
        coco = json.load(f)

    for ann in coco.get("annotations", []):
        ann["category_id"] = 0

    coco["categories"] = [{"id": 0, "name": "tea"}]

    dst_ann.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_ann, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    return {
        "file": str(dst_ann),
        "images": len(coco.get("images", [])),
        "annotations": len(coco.get("annotations", [])),
    }


def main():
    args = parse_args()

    if not args.src.exists():
        raise FileNotFoundError(f"输入数据集不存在：{args.src}")

    if args.dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"输出目录已存在：{args.dst}，如需覆盖请加 --overwrite")
        shutil.rmtree(args.dst)

    shutil.copytree(args.src, args.dst, ignore=shutil.ignore_patterns("annotations"))

    summaries = []
    for ann_rel in [args.train_ann, args.val_ann]:
        summaries.append(merge_annotation_file(args.src / ann_rel, args.dst / ann_rel))

    print(f"输出数据集：{args.dst}")
    for summary in summaries:
        print(
            f"{summary['file']}: "
            f"images={summary['images']}, annotations={summary['annotations']}, categories=[tea]"
        )


if __name__ == "__main__":
    main()
