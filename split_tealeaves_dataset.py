"""
将 TeaLeavesDatasets 物理拆分为左右半图，并重写 COCO 标注。

规则：
- 每张原图拆成 left / right 两张图。
- 普通框按所在半图平移坐标。
- 跨过中线的框只分给主要所在的半图：
  - 左右重叠宽度谁更大，分给谁；
  - 若相等，则按 bbox 中心点在中线左/右判断。
- 分配后 bbox 不切割成两段，只在目标半图坐标系中裁剪到有效边界。

默认不会覆盖原数据集，输出到：
datasets/TeaLeavesDatasets_split_lr
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="拆分 TeaLeavesDatasets 为左右半图 COCO 数据集")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("datasets") / "TeaLeavesDatasets",
        help="原始 TeaLeavesDatasets 根目录",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("datasets") / "TeaLeavesDatasets_split_lr",
        help="输出数据集根目录",
    )
    parser.add_argument("--train_ann", default="annotations/instances_train.json")
    parser.add_argument("--val_ann", default="annotations/instances_val.json")
    parser.add_argument("--overwrite", action="store_true", help="允许删除并重建输出目录")
    return parser.parse_args()


def normalize_file_name(file_name: str) -> Path:
    return Path(file_name.replace("\\", "/"))


def choose_half_for_box(x: float, w: float, mid: float) -> str:
    """跨中线框按主要面积归属；面积相同则按中心点。"""
    x2 = x + w
    left_overlap = max(0.0, min(x2, mid) - max(x, 0.0))
    right_overlap = max(0.0, x2 - max(x, mid))

    if left_overlap > right_overlap:
        return "left"
    if right_overlap > left_overlap:
        return "right"
    return "left" if (x + w / 2.0) < mid else "right"


def remap_bbox_to_half(bbox: list[float], half: str, mid: int, orig_w: int, orig_h: int):
    x, y, w, h = [float(v) for v in bbox]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    if half == "left":
        half_x0 = 0
        half_x1 = mid
        new_x1 = max(0.0, min(x1, float(half_x1)))
        new_x2 = max(0.0, min(x2, float(half_x1)))
    else:
        half_x0 = mid
        half_x1 = orig_w
        new_x1 = max(float(half_x0), min(x1, float(half_x1))) - half_x0
        new_x2 = max(float(half_x0), min(x2, float(half_x1))) - half_x0

    new_y1 = max(0.0, min(y1, float(orig_h)))
    new_y2 = max(0.0, min(y2, float(orig_h)))

    new_w = max(0.0, new_x2 - new_x1)
    new_h = max(0.0, new_y2 - new_y1)
    if new_w <= 0 or new_h <= 0:
        return None
    return [new_x1, new_y1, new_w, new_h]


def build_annotations_by_image(annotations: list[dict]) -> dict[int, list[dict]]:
    ann_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)
    return ann_by_image


def split_one_coco(src_root: Path, dst_root: Path, ann_rel: str, next_image_id: int, next_ann_id: int):
    ann_path = src_root / ann_rel
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    ann_by_image = build_annotations_by_image(coco.get("annotations", []))
    new_images = []
    new_annotations = []
    image_id_map: dict[tuple[int, str], int] = {}

    for img_info in coco["images"]:
        src_rel = normalize_file_name(img_info["file_name"])
        src_img_path = src_root / src_rel
        if not src_img_path.exists():
            raise FileNotFoundError(f"找不到图片：{src_img_path}")

        with Image.open(src_img_path) as image:
            image = image.convert("RGB")
            orig_w, orig_h = image.size
            mid = orig_w // 2

            crops = {
                "left": image.crop((0, 0, mid, orig_h)),
                "right": image.crop((mid, 0, orig_w, orig_h)),
            }

            stem = src_rel.stem
            suffix = src_rel.suffix
            for half, crop in crops.items():
                new_file_name = Path("images") / f"{stem}_{half}{suffix}"
                dst_img_path = dst_root / new_file_name
                dst_img_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(dst_img_path)

                new_img_id = next_image_id
                next_image_id += 1
                image_id_map[(img_info["id"], half)] = new_img_id

                new_images.append({
                    "id": new_img_id,
                    "width": crop.width,
                    "height": crop.height,
                    "file_name": str(new_file_name).replace("/", "\\"),
                })

        for ann in ann_by_image.get(img_info["id"], []):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            half = choose_half_for_box(float(x), float(w), float(mid))
            new_bbox = remap_bbox_to_half(ann["bbox"], half, mid, orig_w, orig_h)
            if new_bbox is None:
                continue

            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            next_ann_id += 1
            new_ann["image_id"] = image_id_map[(img_info["id"], half)]
            new_ann["bbox"] = new_bbox
            new_ann["area"] = new_bbox[2] * new_bbox[3]
            new_ann["segmentation"] = []
            new_annotations.append(new_ann)

    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco.get("categories", []),
    }
    if "licenses" in coco:
        new_coco["licenses"] = coco["licenses"]
    if "info" in coco:
        new_coco["info"] = coco["info"]

    dst_ann_path = dst_root / ann_rel
    dst_ann_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_ann_path, "w", encoding="utf-8") as f:
        json.dump(new_coco, f, ensure_ascii=False, indent=2)

    return {
        "ann": ann_rel,
        "src_images": len(coco["images"]),
        "dst_images": len(new_images),
        "src_annotations": len(coco.get("annotations", [])),
        "dst_annotations": len(new_annotations),
        "next_image_id": next_image_id,
        "next_ann_id": next_ann_id,
    }


def main():
    args = parse_args()
    src_root = args.src
    dst_root = args.dst

    if not src_root.exists():
        raise FileNotFoundError(f"源数据集不存在：{src_root}")

    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"输出目录已存在：{dst_root}，如需覆盖请加 --overwrite")
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    next_image_id = 1
    next_ann_id = 1
    summaries = []
    for ann_rel in [args.train_ann, args.val_ann]:
        summary = split_one_coco(src_root, dst_root, ann_rel, next_image_id, next_ann_id)
        summaries.append(summary)
        next_image_id = summary["next_image_id"]
        next_ann_id = summary["next_ann_id"]

    print(f"输出数据集：{dst_root}")
    for summary in summaries:
        print(
            f"{summary['ann']}: "
            f"images {summary['src_images']} -> {summary['dst_images']}, "
            f"annotations {summary['src_annotations']} -> {summary['dst_annotations']}"
        )


if __name__ == "__main__":
    main()
