"""
训练数据增强等级 1–5（默认 5 = 完整 detection 流水线 + 可选 Mosaic，默认 p=0.3）。

用户可调默认概率/等级见 configs/train.py（AUG_LEVEL、AUG_DET_* 等）；本模块提供等级解析与流水线参数计算。

| 等级 | 说明 |
|------|------|
| 1 | 无增强 |
| 2 | 原图：水平翻转 + ColorJitter（原 simple） |
| 3 | detection 流水线，概率×0.5、光度幅度×0.5 |
| 4 | detection 流水线，概率×0.75、光度幅度×0.75 |
| 5 | detection 全强度（含 RandomPhotometricDistort 色彩）+ 可选 Mosaic（见 configs/augmentation.py） |

旧名：none→1，simple/normal/flip→2，detection_lite→3，detection/detector→5。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TVF
from torchvision.transforms import v2
from transformers import Deimv2ForObjectDetection

from utils.common import PROJECT_ROOT
from configs import preprocess as pc
from utils.preprocess import (
    letterbox_inverse_xyxy,
    letterbox_params,
    letterbox_pil_coco,
    uses_letterbox,
)

AUG_LEVELS = (1, 2, 3, 4, 5)
DEFAULT_AUG_LEVEL = 5

# 等级 3/4 相对等级 5 的检测增强强度（概率与光度幅度同乘）
LEVEL_DETECTION_STRENGTH: dict[int, float] = {3: 0.5, 4: 0.75, 5: 1.0}

_LEGACY_AUG_NAMES: dict[str, int] = {
    "none": 1,
    "simple": 2,
    "normal": 2,
    "flip": 2,
    "detection_lite": 3,
    "detection": 5,
    "detector": 5,
}

SIMPLE_COLOR_JITTER = "brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02"

LEVEL_SUMMARY: dict[int, str] = {
    1: "训练集无增强，原图进 processor",
    2: f"原图分辨率：水平翻转 + ColorJitter（{SIMPLE_COLOR_JITTER}）",
    3: "detection 流水线（概率与光度幅度均为等级 5 的 50%），无 Mosaic",
    4: "detection 流水线（概率与光度幅度均为等级 5 的 75%），无 Mosaic",
    5: "detection 全强度（RandomPhotometricDistort 光度/色彩）+ 可选 Mosaic（见 configs/augmentation.py）",
}

DETECTION_PIPELINE_FIXED = (
    "RandomPhotometricDistort",
    "RandomZoomOut",
    "RandomIoUCrop (RandomApply)",
    "SanitizeBoundingBoxes(min_size=1)",
    "RandomHorizontalFlip",
    "Resize(640×640)",
    "SanitizeBoundingBoxes(min_size=1)",
)

MOSAIC_FIXED = "四宫格 1280×1280 → 缩放到 640×640（仅等级 5 且 train len≥4）"


@dataclass(frozen=True)
class DetectionAugParams:
    photometric_p: float
    zoomout_fill: float
    zoomout_p: float
    iou_crop_p: float
    flip_p: float
    photometric_magnitude: float


def parse_aug_level(value: str | int) -> int:
    if isinstance(value, bool):
        raise ValueError("aug_level 不能为 bool")
    if isinstance(value, int):
        level = value
    else:
        s = str(value).strip().lower()
        if s.isdigit():
            level = int(s)
        elif s in _LEGACY_AUG_NAMES:
            level = _LEGACY_AUG_NAMES[s]
        else:
            raise ValueError(
                f"未知 aug_level={value!r}，可选整数 {AUG_LEVELS} 或旧名: {sorted(_LEGACY_AUG_NAMES)}"
            )
    if level not in AUG_LEVELS:
        raise ValueError(f"aug_level 须在 {AUG_LEVELS} 内，收到 {level}")
    return level


def aug_mode_for_level(level: int) -> str:
    level = parse_aug_level(level)
    if level == 1:
        return "none"
    if level == 2:
        return "simple"
    return "detection"


def uses_detection_pipeline(level: int) -> bool:
    return parse_aug_level(level) >= 3


def uses_mosaic(level: int) -> bool:
    return parse_aug_level(level) == 5


def detection_strength(level: int) -> float:
    level = parse_aug_level(level)
    if level < 3:
        return 0.0
    return LEVEL_DETECTION_STRENGTH[level]


def resolve_detection_params(
    level: int,
    *,
    photometric_p: float,
    zoomout_fill: float,
    zoomout_p: float,
    iou_crop_p: float,
    flip_p: float,
) -> DetectionAugParams:
    """按等级缩放 detection 各步概率；光度幅度用 photometric_magnitude 传入 build。"""
    s = detection_strength(level)
    return DetectionAugParams(
        photometric_p=photometric_p * s,
        zoomout_fill=zoomout_fill,
        zoomout_p=zoomout_p * s,
        iou_crop_p=iou_crop_p * s,
        flip_p=flip_p * s,
        photometric_magnitude=s,
    )


def resolve_train_mosaic_p(level: int, *, mosaic_p: float) -> float:
    """等级 5 使用 mosaic_p；其余等级为 0。"""
    return float(mosaic_p) if uses_mosaic(level) else 0.0


def augmentation_metrics_block(
    level: int,
    *,
    simple_flip_p: float,
    simple_color_p: float,
    det_photometric_p: float,
    det_zoomout_fill: float,
    det_zoomout_p: float,
    det_iou_crop_p: float,
    det_flip_p: float,
    mosaic_p: float,
) -> dict[str, Any]:
    level = parse_aug_level(level)
    mode = aug_mode_for_level(level)
    eff_mosaic_p = resolve_train_mosaic_p(level, mosaic_p=mosaic_p)
    det_eff = (
        resolve_detection_params(
            level,
            photometric_p=det_photometric_p,
            zoomout_fill=det_zoomout_fill,
            zoomout_p=det_zoomout_p,
            iou_crop_p=det_iou_crop_p,
            flip_p=det_flip_p,
        )
        if uses_detection_pipeline(level)
        else None
    )
    block: dict[str, Any] = {
        "level": level,
        "preset": mode,
        "summary": LEVEL_SUMMARY[level],
        "detection_strength": detection_strength(level) if det_eff else None,
        "simple": {
            "flip_p": simple_flip_p,
            "color_p": simple_color_p,
            "color_jitter": SIMPLE_COLOR_JITTER,
        },
        "detection": {
            "photometric_p": det_eff.photometric_p if det_eff else det_photometric_p,
            "photometric_magnitude": det_eff.photometric_magnitude if det_eff else None,
            "zoomout_fill": det_zoomout_fill,
            "zoomout_p": det_eff.zoomout_p if det_eff else det_zoomout_p,
            "iou_crop_p": det_eff.iou_crop_p if det_eff else det_iou_crop_p,
            "flip_p": det_eff.flip_p if det_eff else det_flip_p,
            "mosaic_p": eff_mosaic_p,
            "pipeline_fixed": list(DETECTION_PIPELINE_FIXED),
            "mosaic_note": MOSAIC_FIXED if uses_mosaic(level) and eff_mosaic_p > 0 else None,
        },
    }
    if level == 1:
        block["active_params"] = []
    elif level == 2:
        block["active_params"] = {
            "flip_p": simple_flip_p,
            "color_p": simple_color_p,
        }
    else:
        assert det_eff is not None
        block["active_params"] = {
            "strength": detection_strength(level),
            "photometric_p": det_eff.photometric_p,
            "photometric_magnitude": det_eff.photometric_magnitude,
            "zoomout_fill": det_zoomout_fill,
            "zoomout_p": det_eff.zoomout_p,
            "iou_crop_p": det_eff.iou_crop_p,
            "flip_p": det_eff.flip_p,
            "mosaic_p": eff_mosaic_p,
        }
    return block


def _fmt_p(v: object) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _aug_params(aug: dict) -> dict:
    active = aug.get("active_params")
    if isinstance(active, dict) and active:
        return active
    level = aug.get("level")
    if level is None and "preset" in aug:
        try:
            level = parse_aug_level(aug["preset"])
        except ValueError:
            level = None
    if level == 2:
        return aug.get("simple") or {}
    if level is not None and int(level) >= 3:
        return aug.get("detection") or {}
    preset = aug.get("preset")
    if preset == "simple":
        return aug.get("simple") or {}
    if preset == "detection":
        return aug.get("detection") or {}
    return {}


def format_augmentation_summary(aug: dict | None) -> list[str]:
    if not aug:
        return ["数据增强: —"]

    level = aug.get("level")
    if level is None:
        try:
            level = parse_aug_level(aug.get("preset", 1))
        except ValueError:
            level = "?"
    if level == 1:
        return [f"数据增强[L{level}]"]

    params = _aug_params(aug)
    lv = int(level) if isinstance(level, int) else level

    if lv == 2:
        parts = [
            f"HFlip {_fmt_p(params.get('flip_p', '—'))}",
            f"ColorJitter {_fmt_p(params.get('color_p', '—'))}",
        ]
        return [f"数据增强[L{lv}]: {', '.join(parts)}"]

    if isinstance(lv, int) and lv >= 3:
        strength = params.get("strength", params.get("photometric_magnitude", "—"))
        parts = [
            f"强度×{_fmt_p(strength)}",
            f"Photometric {_fmt_p(params.get('photometric_p', '—'))}",
            f"ZoomOut {_fmt_p(params.get('zoomout_p', '—'))}",
            f"IoUCrop {_fmt_p(params.get('iou_crop_p', '—'))}",
            f"HFlip {_fmt_p(params.get('flip_p', '—'))}",
            f"Mosaic {_fmt_p(params.get('mosaic_p', '—'))}",
        ]
        return [f"数据增强[L{lv}]: {', '.join(parts)}"]

    return [f"数据增强[{level}]"]


def format_augmentation_log_line(aug: dict) -> str:
    return format_augmentation_summary(aug)[0]


# ========================================================================
# Train Coco
# ========================================================================

def resolve_cli_path(path: Path, *, script_dir: Path | None = None) -> Path:
    """相对路径相对于项目根目录解析。"""
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    base = script_dir or PROJECT_ROOT
    return (base / path).resolve()


def _default_coco_train_val_paths(root: Path) -> tuple[Path, Path]:
    """返回 (train_ann, val_ann)。支持 LabelMe 风格 train.json/val.json 或 COCO instances_*.json。"""
    ann = root / "annotations"
    a1, b1 = ann / "train.json", ann / "val.json"
    if a1.is_file() and b1.is_file():
        return a1, b1
    a2, b2 = ann / "instances_train.json", ann / "instances_val.json"
    if a2.is_file() and b2.is_file():
        return a2, b2
    raise FileNotFoundError(
        f"在 {ann} 下未找到 train.json+val.json 或 instances_train.json+instances_val.json（数据集根: {root}）"
    )


@dataclass(frozen=True)
class DatasetCategorySpec:
    """由 --datasets 下各集 train 标注汇总得到的检测类别。"""

    num_labels: int
    id2label: dict[int, str]
    label2id: dict[str, int]
    coco_id_to_label: dict[int, int]
    label_to_coco_id: dict[int, int]


def config_num_labels(config) -> int:
    nl = getattr(config, "num_labels", None)
    if nl is not None:
        return int(nl)
    id2label = getattr(config, "id2label", None) or {}
    return len(id2label)


def resolve_categories_from_dataset_roots(roots: list[Path]) -> DatasetCategorySpec:
    """
    扫描 --datasets 下各集 train 标注，合并全部类别（按 COCO category_id 去重），
    num_labels = 类别集合大小；训练时将 category_id 映射为连续 label 0..num_labels-1，
    mAP 评测时再映射回 COCO category_id。
    """
    if not roots:
        raise ValueError("resolve_categories_from_dataset_roots 需要至少一个数据集根目录")

    categories_meta: dict[int, str] = {}
    ann_cat_ids: set[int] = set()

    for root in roots:
        root = root.resolve()
        train_p, _ = _default_coco_train_val_paths(root)
        with open(train_p, encoding="utf-8") as f:
            coco = json.load(f)
        for cat in coco.get("categories", []):
            cid = int(cat["id"])
            name = str(cat.get("name", cid))
            if cid in categories_meta and categories_meta[cid] != name:
                print(
                    f"警告: category_id={cid} 在数据集 {root.name} 中名称与先前不一致："
                    f"{categories_meta[cid]!r} vs {name!r}，保留先出现的名称。"
                )
            categories_meta.setdefault(cid, name)
        for ann in coco.get("annotations", []):
            ann_cat_ids.add(int(ann["category_id"]))

    if ann_cat_ids:
        missing = ann_cat_ids - set(categories_meta)
        for cid in sorted(missing):
            categories_meta[cid] = str(cid)
            print(
                f"警告: category_id={cid} 出现在 train 标注中但 categories 未定义，"
                f"使用占位名称 {cid!r}。"
            )

    if not categories_meta:
        roots_s = ", ".join(str(r) for r in roots)
        raise ValueError(f"在以下数据集的 train 标注中未找到任何类别: {roots_s}")

    all_class_ids = set(categories_meta.keys())
    sorted_coco_ids = sorted(all_class_ids)
    num_labels = len(all_class_ids)
    coco_id_to_label = {cid: i for i, cid in enumerate(sorted_coco_ids)}
    label_to_coco_id = {i: cid for cid, i in coco_id_to_label.items()}
    id2label = {i: categories_meta[cid] for i, cid in enumerate(sorted_coco_ids)}
    label2id = {name: i for i, name in id2label.items()}

    return DatasetCategorySpec(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        coco_id_to_label=coco_id_to_label,
        label_to_coco_id=label_to_coco_id,
    )


def _remap_target_category_ids(target: list, category_id_remap: dict[int, int]) -> list:
    if not category_id_remap:
        return target
    out: list = []
    for ann in target:
        a = dict(ann)
        raw_cid = a["category_id"]
        if isinstance(raw_cid, torch.Tensor):
            cid = int(raw_cid.item())
        else:
            cid = int(raw_cid)
        if cid not in category_id_remap:
            raise ValueError(
                f"标注 category_id={cid} 不在类别表内，已知 COCO id: {sorted(category_id_remap)}"
            )
        a["category_id"] = category_id_remap[cid]
        out.append(a)
    return out


def _abs_image_file_name(root: Path, file_name: str) -> str:
    p = Path(file_name)
    if p.is_absolute():
        return str(p.resolve())
    return str((root / file_name).resolve())


def _resolve_dataset_ratios(ratios: list[float] | None, n_roots: int) -> list[float] | None:
    """多集训练时返回长度为 n_roots 的正比例；单集或未指定时返回 None。"""
    if n_roots <= 1:
        return None
    if ratios is None:
        return [1.0] * n_roots
    if len(ratios) != n_roots:
        raise ValueError(f"--dataset_ratios 数量({len(ratios)})须与 --datasets({n_roots}) 一致")
    if any(r <= 0 for r in ratios):
        raise ValueError("--dataset_ratios 各项须 > 0")
    return [float(r) for r in ratios]


def merge_coco_roots_for_training(
    roots: list[Path],
    cache_dir: Path,
    *,
    dataset_ratios: list[float] | None = None,
) -> tuple[Path, Path, list[float]]:
    """
    将多个 COCO 根目录的 train / val 各合并为一份 JSON，写入 cache_dir。
    - images[].file_name 改为绝对路径，便于单根目录占位 + torchvision CocoDetection 加载。
    - image_id、annotation id 全局连续重编号；category_id 保持原样（各数据集须兼容同一套 id）。
    返回 (merged_train_json_path, merged_val_json_path, train_sample_weights)。
    train_sample_weights[i] = ratio_k / n_k（第 i 张图属于第 k 个源集），供 WeightedRandomSampler 使用。
    """
    if len(roots) < 2:
        raise ValueError("merge_coco_roots_for_training 至少需要 2 个根目录")

    ratios = _resolve_dataset_ratios(dataset_ratios, len(roots))
    assert ratios is not None

    cache_dir.mkdir(parents=True, exist_ok=True)
    categories_merged: dict[int, dict] = {}
    train_sample_weights: list[float] = []

    def merge_split(split_key: str, out_path: Path) -> tuple[int, int]:
        nonlocal train_sample_weights
        images_out: list[dict] = []
        anns_out: list[dict] = []
        next_img_id = 1
        next_ann_id = 1
        if split_key == "train":
            train_sample_weights = []

        for root_idx, root in enumerate(roots):
            root = root.resolve()
            train_p, val_p = _default_coco_train_val_paths(root)
            ann_path = train_p if split_key == "train" else val_p
            with open(ann_path, "r", encoding="utf-8") as f:
                coco = json.load(f)

            for cat in coco.get("categories", []):
                cid = int(cat["id"])
                if cid in categories_merged and categories_merged[cid].get("name") != cat.get("name"):
                    print(
                        f"警告: category_id={cid} 在不同数据集中名称不一致："
                        f"{categories_merged[cid].get('name')!r} vs {cat.get('name')!r}，保留先出现的定义。"
                    )
                categories_merged.setdefault(cid, dict(cat))

            id_old_to_new: dict[int, int] = {}
            imgs_this_root: list[dict] = []
            for img in coco.get("images", []):
                old_iid = int(img["id"])
                if old_iid in id_old_to_new:
                    raise ValueError(f"数据集 {root} 的 {split_key} 中存在重复 image id: {old_iid}")
                nid = next_img_id
                next_img_id += 1
                id_old_to_new[old_iid] = nid
                new_img = dict(img)
                new_img["id"] = nid
                fn = img.get("file_name")
                if not fn:
                    raise ValueError(f"image id={old_iid} 缺少 file_name（{ann_path}）")
                new_img["file_name"] = _abs_image_file_name(root, str(fn))
                imgs_this_root.append(new_img)

            images_out.extend(imgs_this_root)
            if split_key == "train":
                n_i = len(imgs_this_root)
                w = ratios[root_idx] / max(n_i, 1)
                train_sample_weights.extend([w] * n_i)

            for ann in coco.get("annotations", []):
                oid = int(ann["image_id"])
                if oid not in id_old_to_new:
                    continue
                new_ann = {k: v for k, v in ann.items() if k not in ("id", "image_id")}
                new_ann["id"] = next_ann_id
                next_ann_id += 1
                new_ann["image_id"] = id_old_to_new[oid]
                anns_out.append(new_ann)

        cats_list = [categories_merged[k] for k in sorted(categories_merged.keys())]
        merged = {
            "images": images_out,
            "annotations": anns_out,
            "categories": cats_list,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False)
        return len(images_out), len(anns_out)

    train_out = cache_dir / "merged_train.json"
    val_out = cache_dir / "merged_val.json"
    n_tr_img, n_tr_ann = merge_split("train", train_out)
    n_va_img, n_va_ann = merge_split("val", val_out)
    ratio_str = ":".join(f"{r:g}" for r in ratios)
    print(
        f"多数据集合并: {len(roots)} 个根目录 ->\n  训练标注 {train_out}\n  验证标注 {val_out}\n"
        f"  train 图像={n_tr_img}, 标注={n_tr_ann}; val 图像={n_va_img}, 标注={n_va_ann}\n"
        f"  train 采样比例 {ratio_str}（WeightedRandomSampler）"
    )
    return train_out, val_out, train_sample_weights


# ========================================================================
# Train Augment
# ========================================================================

def _coco_anns_to_tv_target(annotations: list[dict], *, image_height: int, image_width: int) -> dict:
    """COCO 标注列表 -> v2 所需的 boxes (XYXY) + labels。canvas_size 为 (H, W)。"""
    boxes_list: list[list[float]] = []
    labels_list: list[int] = []
    for t in annotations:
        if t.get("iscrowd", 0) == 1:
            continue
        x, y, bw, bh = (float(v) for v in t["bbox"])
        if bw <= 0 or bh <= 0:
            continue
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        x1 = max(0.0, min(x1, float(image_width)))
        x2 = max(0.0, min(x2, float(image_width)))
        y1 = max(0.0, min(y1, float(image_height)))
        y2 = max(0.0, min(y2, float(image_height)))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue
        boxes_list.append([x1, y1, x2, y2])
        labels_list.append(int(t["category_id"]))
    if not boxes_list:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
    else:
        boxes = torch.tensor(boxes_list, dtype=torch.float32)
        labels = torch.tensor(labels_list, dtype=torch.int64)
    h, w = int(image_height), int(image_width)
    tv_boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w))
    return {"boxes": tv_boxes, "labels": labels}


def _tv_target_to_coco_anns(target: dict) -> list[dict]:
    """v2 输出 target -> COCO 风格 annotations（供 HF processor）。"""
    boxes = target["boxes"]
    arr = boxes.cpu()
    labs = target["labels"].cpu()
    out: list[dict] = []
    for i in range(arr.shape[0]):
        x1, y1, x2, y2 = (float(v) for v in arr[i].tolist())
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        out.append(
            {
                "category_id": int(labs[i].item()),
                "bbox": [x1, y1, bw, bh],
                "iscrowd": 0,
                "area": float(bw * bh),
                "segmentation": [],
            }
        )
    return out


def build_detection_augment(
    *,
    photometric_p: float,
    zoomout_fill: int | float,
    zoomout_p: float,
    iou_crop_p: float,
    flip_p: float,
    photometric_magnitude: float = 1.0,
    final_stretch_resize: bool = True,
) -> v2.Compose:
    """对齐常见 RT-DETR/DEIM 式 train dataloader；letterbox 模式下不在此做末尾 Resize。"""
    m = max(0.0, min(1.0, float(photometric_magnitude)))
    photometric_steps: list = []
    if photometric_p > 0:
        if m >= 0.999:
            photometric_steps.append(v2.RandomPhotometricDistort(p=photometric_p))
        else:
            # 等级 3/4：同流水线，光度用较弱 ColorJitter 近似减小幅度
            photometric_steps.append(
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.2 * m,
                            contrast=0.2 * m,
                            saturation=0.2 * m,
                            hue=max(0.0, 0.02 * m),
                        )
                    ],
                    p=photometric_p,
                )
            )
    steps: list = [
        *photometric_steps,
        v2.RandomZoomOut(fill=zoomout_fill, p=zoomout_p),
        v2.RandomApply([v2.RandomIoUCrop()], p=iou_crop_p),
        v2.SanitizeBoundingBoxes(min_size=1.0),
        v2.RandomHorizontalFlip(p=flip_p),
    ]
    if final_stretch_resize:
        steps.extend([v2.Resize(size=(640, 640)), v2.SanitizeBoundingBoxes(min_size=1.0)])
    return v2.Compose(steps)


def _scalar_ann_field(v):
    if isinstance(v, torch.Tensor):
        return int(v.item()) if v.numel() == 1 else v.tolist()
    return v


def _mosaic_pil_coco_quadrants(
    quadrants: list[tuple[Image.Image, list]],
) -> tuple[Image.Image, list[dict]]:
    """
    四宫格 Mosaic：画布 1280×1280，每格将原图拉伸到 640×640 后粘贴，再整体缩放到 640×640。
    标注为 COCO xywh，与 torchvision CocoDetection 一致。
    """
    assert len(quadrants) == 4
    qsize = 640
    wc, hc = 1280, 1280
    canvas = Image.new("RGB", (wc, hc), (114, 114, 114))
    merged: list[dict] = []
    corners = [(0, 0), (qsize, 0), (0, qsize), (qsize, qsize)]
    for (img, anns), (ox, oy) in zip(quadrants, corners):
        w0, h0 = img.size
        if w0 <= 0 or h0 <= 0:
            continue
        img_r = img.resize((qsize, qsize), Image.BILINEAR)
        canvas.paste(img_r, (ox, oy))
        sx = qsize / float(w0)
        sy = qsize / float(h0)
        for ann in anns:
            if int(_scalar_ann_field(ann.get("iscrowd", 0))) == 1:
                continue
            bbox = ann.get("bbox")
            if bbox is None:
                continue
            if isinstance(bbox, torch.Tensor):
                bx = [float(t) for t in bbox.flatten().tolist()]
            else:
                bx = [float(t) for t in bbox]
            if len(bx) != 4:
                continue
            x, y, bw, bh = bx
            if bw <= 0 or bh <= 0:
                continue
            x1 = x * sx + ox
            y1 = y * sy + oy
            x2 = (x + bw) * sx + ox
            y2 = (y + bh) * sy + oy
            x1 = max(0.0, min(x1, float(wc - 1)))
            y1 = max(0.0, min(y1, float(hc - 1)))
            x2 = max(0.0, min(x2, float(wc)))
            y2 = max(0.0, min(y2, float(hc)))
            bw2, bh2 = x2 - x1, y2 - y1
            if bw2 <= 1.0 or bh2 <= 1.0:
                continue
            cid = ann["category_id"]
            if isinstance(cid, torch.Tensor):
                cid = int(cid.item())
            else:
                cid = int(cid)
            merged.append(
                {
                    "category_id": cid,
                    "bbox": [x1, y1, bw2, bh2],
                    "iscrowd": 0,
                    "segmentation": [],
                    "area": float(bw2 * bh2),
                }
            )

    out = canvas.resize((640, 640), Image.BILINEAR)
    s = 0.5
    merged_out: list[dict] = []
    for a in merged:
        x, y, bw, bh = a["bbox"]
        merged_out.append(
            {
                "category_id": a["category_id"],
                "bbox": [x * s, y * s, bw * s, bh * s],
                "iscrowd": 0,
                "segmentation": [],
                "area": float(max(bw * s, 0.0) * max(bh * s, 0.0)),
            }
        )
    return out, merged_out


def _collate_processor_batch(images, annotations, processor):
    encoding = processor(images=images, annotations=annotations, return_tensors="pt")
    labels = [{k: v for k, v in lab.items()} for lab in encoding["labels"]]
    batch_out = {
        "pixel_values": encoding["pixel_values"],
        "labels": labels,
    }
    if "pixel_mask" in encoding:
        batch_out["pixel_mask"] = encoding["pixel_mask"]
    return batch_out


def _apply_train_augment_simple(
    image: Image.Image,
    annotations: list[dict],
    *,
    flip_p: float,
    color_p: float,
    color_jitter: T.ColorJitter,
) -> tuple[Image.Image, list[dict]]:
    """原图分辨率：水平翻转 + ColorJitter（等级 2）。"""
    anns = []
    for a in annotations:
        b = dict(a)
        if "bbox" in b:
            b["bbox"] = [float(x) for x in b["bbox"]]
        anns.append(b)

    w, h = image.size
    if flip_p > 0 and torch.rand(()).item() < flip_p:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        for obj in anns:
            if "bbox" not in obj:
                continue
            x, y, bw, bh = obj["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            obj["bbox"] = [float(w - x - bw), float(y), float(bw), float(bh)]

    if color_p > 0 and torch.rand(()).item() < color_p:
        image = color_jitter(image)

    return image, anns


def make_train_collate_fn(
    processor,
    *,
    aug_level: int,
    simple_flip_p: float,
    simple_color_p: float,
    det_photometric_p: float,
    det_zoomout_fill: int | float,
    det_zoomout_p: float,
    det_iou_crop_p: float,
    det_flip_p: float,
):
    aug_level = parse_aug_level(aug_level)
    mode = aug_mode_for_level(aug_level)
    color_jitter = T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02)
    detection_tf: v2.Compose | None = None
    det_params = None
    if uses_detection_pipeline(aug_level):
        det_params = resolve_detection_params(
            aug_level,
            photometric_p=det_photometric_p,
            zoomout_fill=det_zoomout_fill,
            zoomout_p=det_zoomout_p,
            iou_crop_p=det_iou_crop_p,
            flip_p=det_flip_p,
        )
        detection_tf = build_detection_augment(
            photometric_p=det_params.photometric_p,
            zoomout_fill=det_params.zoomout_fill,
            zoomout_p=det_params.zoomout_p,
            iou_crop_p=det_params.iou_crop_p,
            flip_p=det_params.flip_p,
            photometric_magnitude=det_params.photometric_magnitude,
            final_stretch_resize=not uses_letterbox(),
        )

    def collate(batch):
        images = []
        annotations = []
        for img, image_id, target in batch:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tgt = list(target)
            if mode == "simple":
                img, tgt = _apply_train_augment_simple(
                    img, tgt, flip_p=simple_flip_p, color_p=simple_color_p, color_jitter=color_jitter
                )
            elif mode == "detection":
                assert detection_tf is not None
                w, h = img.size
                im_t = v2.functional.to_image(img)
                tdict = _coco_anns_to_tv_target(tgt, image_height=h, image_width=w)
                im_t2, t2 = detection_tf(im_t, tdict)
                img = TVF.to_pil_image(im_t2)
                tgt = _tv_target_to_coco_anns(t2)
            elif mode == "none":
                pass
            else:
                raise ValueError(f"未知 aug_level 内部模式: {mode}")
            if uses_letterbox():
                img, tgt = letterbox_pil_coco(img, tgt)
            images.append(img)
            annotations.append({"image_id": image_id, "annotations": tgt})
        return _collate_processor_batch(images, annotations, processor)

    return collate


class TeaCocoDataset(torch.utils.data.Dataset):
    """COCO 检测；训练集可设 mosaic_p 在四张图间随机 Mosaic（需 len>=4）。"""

    def __init__(
        self,
        root: Path,
        ann_file: Path,
        *,
        mosaic_p: float = 0.0,
        sample_weights: list[float] | None = None,
        category_id_remap: dict[int, int] | None = None,
    ):
        self._coco = CocoDetection(str(root), str(ann_file))
        self.mosaic_p = float(mosaic_p)
        self.sample_weights = sample_weights
        self._category_id_remap = category_id_remap or {}

    def _remap_target(self, target: list) -> list:
        return _remap_target_category_ids(target, self._category_id_remap)

    def __len__(self):
        return len(self._coco)

    def __getitem__(self, idx: int):
        if self.mosaic_p > 0.0 and len(self._coco) >= 4 and torch.rand(()).item() < self.mosaic_p:
            return self._getitem_mosaic(idx)
        img, target = self._coco[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")
        image_id = self._coco.ids[idx]
        return img, image_id, self._remap_target(list(target))

    def _getitem_mosaic(self, idx: int) -> tuple[Image.Image, int, list]:
        n = len(self._coco)
        others: list[int] = []
        seen = {idx}
        for _ in range(n * 4 + 10):
            j = int(torch.randint(0, n, (1,)).item())
            if j not in seen:
                others.append(j)
                seen.add(j)
                if len(others) == 3:
                    break
        if len(others) < 3:
            img, target = self._coco[idx]
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img, self._coco.ids[idx], self._remap_target(list(target))

        idxs = [idx, others[0], others[1], others[2]]
        quads: list[tuple[Image.Image, list]] = []
        for j in idxs:
            img, target = self._coco[j]
            if img.mode != "RGB":
                img = img.convert("RGB")
            quads.append((img, list(target)))
        mos_img, mos_tgt = _mosaic_pil_coco_quadrants(quads)
        return mos_img, self._coco.ids[idx], self._remap_target(mos_tgt)


# ========================================================================
# Train Model
# ========================================================================

def freeze_all_except_classification_heads(model: Deimv2ForObjectDetection) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(
            k in name
            for k in (
                "class_embed",
                "enc_score_head",
                "denoising_class_embed",
            )
        ):
            p.requires_grad = True


def _is_conv_encoder_backbone_param(name: str) -> bool:
    """DINOv3：conv_encoder.backbone；CNN 骨干：conv_encoder.model（不含 encoder_input_proj）。"""
    return ".conv_encoder.backbone." in name or ".conv_encoder.model." in name


def freeze_backbone_only(model: Deimv2ForObjectDetection) -> None:
    """只冻结 conv_encoder 内的视觉骨干，其余（neck、encoder、decoder、头等）全部训练。"""
    for name, p in model.named_parameters():
        p.requires_grad = not _is_conv_encoder_backbone_param(name)


_DINOV3_BACKBONE_LAYER_IDX = re.compile(r"\.conv_encoder\.backbone\.model\.layer\.(\d+)\.")


def apply_dinov3_backbone_last_n_unfreeze(model: Deimv2ForObjectDetection, n: int) -> None:
    """在 apply_train_mode 之后调用：仅当骨干为 DINOv3 ViT 时，解冻最后 n 个 encoder block 及 backbone 末端 LayerNorm。"""
    if n <= 0:
        return
    bb_cfg = getattr(model.config, "backbone_config", None)
    if bb_cfg is None or getattr(bb_cfg, "model_type", None) != "dinov3_vit":
        print(
            "警告: --unfreeze_backbone_last_n>0，但当前模型 backbone_config 不是 dinov3_vit，已跳过骨干局部解冻。"
        )
        return
    num_layers = int(getattr(bb_cfg, "num_hidden_layers", 0) or 0)
    if num_layers <= 0:
        print("警告: 无法读取 backbone num_hidden_layers，已跳过 DINOv3 骨干局部解冻。")
        return

    n_eff = min(int(n), num_layers)
    min_layer = num_layers - n_eff
    newly_trainable_numel = 0

    for name, p in model.named_parameters():
        if ".conv_encoder.backbone.norm." in name:
            if not p.requires_grad:
                newly_trainable_numel += p.numel()
            p.requires_grad = True
            continue
        m = _DINOV3_BACKBONE_LAYER_IDX.search(name)
        if m is None:
            continue
        if int(m.group(1)) < min_layer:
            continue
        if not p.requires_grad:
            newly_trainable_numel += p.numel()
        p.requires_grad = True

    print(
        f"DINOv3 骨干：已解冻最后 {n_eff} 个 Transformer block（layer_index>={min_layer}）"
        f"及 conv_encoder.backbone.norm；新增可训练参数约 {newly_trainable_numel:,} 个元素。"
    )


def _dinov3_vit_num_hidden_layers(model: Deimv2ForObjectDetection) -> int | None:
    bb_cfg = getattr(model.config, "backbone_config", None)
    if bb_cfg is None or getattr(bb_cfg, "model_type", None) != "dinov3_vit":
        return None
    n = int(getattr(bb_cfg, "num_hidden_layers", 0) or 0)
    return n if n > 0 else None


def _param_no_weight_decay(name: str) -> bool:
    """与常见检测配置一致：LayerNorm / BN / bias 不做 weight decay。"""
    lname = name.lower()
    if lname.endswith(".bias"):
        return True
    if "norm" in lname or ".bn" in lname or "layernorm" in lname:
        return True
    return False


def _dinov3_backbone_lr_for_block(
    *,
    num_layers: int,
    lr_backbone: float,
    decay_per_block: float,
    block_index: int | None,
    is_final_norm: bool,
) -> float:
    """block_index: None 表示 embeddings / rope 等 stem；is_final_norm 为 backbone 末端 LayerNorm。"""
    L = num_layers
    d = float(decay_per_block)
    lb = float(lr_backbone)
    if is_final_norm:
        return lb
    if block_index is None:
        return lb * (d**L)
    return lb * (d ** (L - 1 - int(block_index)))


def build_adamw_param_groups(
    model: Deimv2ForObjectDetection,
    *,
    lr: float,
    weight_decay: float,
    unfreeze_backbone_last_n: int,
    lr_backbone: float,
    backbone_lr_decay: float,
) -> tuple[list[dict[str, object]], bool]:
    """
    构造 AdamW 的 param_groups。
    当 unfreeze_backbone_last_n>0 且当前为 DINOv3 ViT 骨干时：仅对可训练的 model.conv_encoder.backbone.*
    按 ViT block 分层 lr（浅层更低）；其余（含 conv_encoder.model、STA、fusion_proj、检测头等）使用 lr。
    否则：骨干与检测部分均用单一 lr（与旧脚本行为一致）。
    返回 (param_groups, used_layerwise)。
    """
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        return [{"params": [], "lr": lr, "weight_decay": weight_decay}], False

    num_layers = _dinov3_vit_num_hidden_layers(model)
    want_layerwise = int(unfreeze_backbone_last_n) > 0
    use_layerwise = bool(want_layerwise and num_layers is not None)
    if want_layerwise and num_layers is None:
        print(
            "警告: --unfreeze_backbone_last_n>0 时本可对 DINOv3 骨干启用分层 lr，但当前 backbone_config 不是 dinov3_vit，"
            "已回退为单一 --lr。"
        )

    if not use_layerwise:
        bb = [p for n, p in trainable if _is_conv_encoder_backbone_param(n)]
        rest = [p for n, p in trainable if not _is_conv_encoder_backbone_param(n)]
        groups: list[dict[str, object]] = []
        if bb:
            groups.append({"params": bb, "lr": lr, "weight_decay": weight_decay})
        if rest:
            groups.append({"params": rest, "lr": lr, "weight_decay": weight_decay})
        if not groups:
            return [{"params": [], "lr": lr, "weight_decay": weight_decay}], False
        return groups, False

    buckets: dict[tuple[float, float], list[torch.nn.Parameter]] = defaultdict(list)

    def add(name: str, p: torch.nn.Parameter, lr_v: float) -> None:
        wd = 0.0 if _param_no_weight_decay(name) else weight_decay
        buckets[(float(lr_v), float(wd))].append(p)

    for name, p in trainable:
        if ".conv_encoder.backbone." in name:
            if ".conv_encoder.backbone.norm." in name:
                lr_v = _dinov3_backbone_lr_for_block(
                    num_layers=num_layers,  # type: ignore[arg-type]
                    lr_backbone=lr_backbone,
                    decay_per_block=backbone_lr_decay,
                    block_index=None,
                    is_final_norm=True,
                )
            elif (
                ".conv_encoder.backbone.embeddings." in name
                or ".conv_encoder.backbone.rope_embeddings." in name
            ):
                lr_v = _dinov3_backbone_lr_for_block(
                    num_layers=num_layers,  # type: ignore[arg-type]
                    lr_backbone=lr_backbone,
                    decay_per_block=backbone_lr_decay,
                    block_index=None,
                    is_final_norm=False,
                )
            else:
                m = _DINOV3_BACKBONE_LAYER_IDX.search(name)
                if m is not None:
                    lr_v = _dinov3_backbone_lr_for_block(
                        num_layers=num_layers,  # type: ignore[arg-type]
                        lr_backbone=lr_backbone,
                        decay_per_block=backbone_lr_decay,
                        block_index=int(m.group(1)),
                        is_final_norm=False,
                    )
                else:
                    lr_v = _dinov3_backbone_lr_for_block(
                        num_layers=num_layers,  # type: ignore[arg-type]
                        lr_backbone=lr_backbone,
                        decay_per_block=backbone_lr_decay,
                        block_index=None,
                        is_final_norm=False,
                    )
            add(name, p, lr_v)
        else:
            add(name, p, lr)

    groups: list[dict[str, object]] = [
        {"params": params, "lr": lr_k, "weight_decay": wd_k}
        for (lr_k, wd_k), params in sorted(buckets.items(), key=lambda x: (x[0][1], x[0][0]))
        if params
    ]
    return groups, True


def sync_optimizer_param_group_metadata(
    opt: AdamW,
    model: Deimv2ForObjectDetection,
    *,
    lr: float,
    weight_decay: float,
    unfreeze_backbone_last_n: int,
    lr_backbone: float,
    backbone_lr_decay: float,
) -> None:
    """根据当前可训练参数与超参，写入各 param_group 的 base_lr / is_backbone（续训加载优化器后须调用）。"""
    fresh_groups, _ = build_adamw_param_groups(
        model,
        lr=lr,
        weight_decay=weight_decay,
        unfreeze_backbone_last_n=unfreeze_backbone_last_n,
        lr_backbone=lr_backbone,
        backbone_lr_decay=backbone_lr_decay,
    )
    pid_to_base: dict[int, float] = {}
    for g in fresh_groups:
        blr = float(g["lr"])
        for p in g["params"]:
            pid_to_base[id(p)] = blr
    id2name = _param_id_to_param_name(model)
    for g in opt.param_groups:
        params = g["params"]
        if not params:
            g["base_lr"] = float(g.get("lr", lr))
            g["is_backbone"] = False
            continue
        bases = [pid_to_base.get(id(p)) for p in params]
        if any(b is None for b in bases):
            g["base_lr"] = float(g.get("lr", lr))
        else:
            g["base_lr"] = float(bases[0])
        names = [id2name.get(id(p), "") for p in params]
        g["is_backbone"] = bool(names) and all(_is_conv_encoder_backbone_param(n) for n in names)


def apply_backbone_linear_warmup_lrs(opt: AdamW, epoch: int, warmup_epochs: int) -> float:
    """按 epoch（1-based）对 is_backbone 组做线性热身：lr = base_lr * mult，mult 从 0 增至 1。非骨干组恒为 base_lr。返回骨干 mult。"""
    if warmup_epochs <= 0:
        for g in opt.param_groups:
            g["lr"] = float(g["base_lr"])
        return 1.0
    if warmup_epochs == 1:
        m_bb = 1.0
    elif epoch <= warmup_epochs:
        m_bb = (epoch - 1) / float(warmup_epochs - 1)
    else:
        m_bb = 1.0
    for g in opt.param_groups:
        base = float(g["base_lr"])
        if g.get("is_backbone"):
            g["lr"] = base * m_bb
        else:
            g["lr"] = base
    return m_bb


def _param_id_to_param_name(model: Deimv2ForObjectDetection) -> dict[int, str]:
    """每个 Parameter 对象 id -> 首次出现的参数名（用于优化器 state 与模型对齐）。"""
    out: dict[int, str] = {}
    for name, p in model.named_parameters():
        out.setdefault(id(p), name)
    return out


def pack_adamw_named_state(opt: torch.optim.Optimizer, model: Deimv2ForObjectDetection) -> dict[str, dict[str, torch.Tensor]]:
    """按参数名导出 AdamW 的 state 张量（CPU），供 param_groups 结构变化时续训合并。"""
    id2name = _param_id_to_param_name(model)
    packed: dict[str, dict[str, torch.Tensor]] = {}
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p)
            if not st:
                continue
            pname = id2name.get(id(p))
            if pname is None:
                continue
            bucket: dict[str, torch.Tensor] = {}
            for k, v in st.items():
                if torch.is_tensor(v):
                    bucket[k] = v.detach().cpu().clone()
            if bucket:
                packed[pname] = bucket
    return packed


def apply_adamw_named_state(opt: torch.optim.Optimizer, model: Deimv2ForObjectDetection, packed: object) -> int:
    """将 pack_adamw_named_state 保存的动量写回当前优化器。返回成功对齐并写入的参数个数。"""
    if not isinstance(packed, dict) or not packed:
        return 0
    name_to_p = dict(model.named_parameters())
    opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    n_ok = 0
    for pname, src in packed.items():
        if pname not in name_to_p:
            continue
        p = name_to_p[pname]
        if id(p) not in opt_param_ids or not p.requires_grad:
            continue
        new_st: dict[str, torch.Tensor] = {}
        ok = True
        for key in ("exp_avg", "exp_avg_sq"):
            if key not in src:
                continue
            t = src[key]
            if not torch.is_tensor(t) or t.shape != p.shape:
                ok = False
                break
            new_st[key] = t.to(device=p.device, dtype=p.dtype).clone()
        if not ok:
            continue
        if "step" in src:
            sv = src["step"]
            if torch.is_tensor(sv):
                new_st["step"] = sv.to(device=p.device).clone()
            elif isinstance(sv, (int, float)):
                new_st["step"] = torch.tensor(float(sv), device=p.device, dtype=torch.float32)
        opt.state[p] = new_st
        n_ok += 1
    return n_ok


def freeze_all_except_detection_heads(model: Deimv2ForObjectDetection) -> None:
    """仅训练各检测头（class / bbox / enc_score / denoising 等），encoder 与 decoder 主体仍冻结。"""
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if any(
            k in name
            for k in (
                "class_embed",
                "enc_score_head",
                "denoising_class_embed",
                "bbox_embed",
            )
        ):
            p.requires_grad = True


def set_classification_only_loss(model: Deimv2ForObjectDetection) -> None:
    model.config.weight_loss_bbox = 0.0
    model.config.weight_loss_giou = 0.0
    model.config.weight_loss_fgl = 0.0
    model.config.weight_loss_ddf = 0.0


def restore_default_deimv2_loss_weights(
    model: Deimv2ForObjectDetection, bbox_scale: float = 1.0
) -> None:
    """与 Transformers Deimv2Config 默认一致；bbox_scale 同时放大 bbox 与 giou 权重。"""
    s = float(bbox_scale)
    model.config.weight_loss_mal = 1.0
    model.config.weight_loss_bbox = 5.0 * s
    model.config.weight_loss_giou = 2.0 * s
    model.config.weight_loss_fgl = 0.15
    model.config.weight_loss_ddf = 1.5


def move_labels_to_device(labels: list[dict], device: torch.device) -> list[dict]:
    out = []
    for lab in labels:
        lab = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in lab.items()}
        out.append(lab)
    return out


def collate_infer_only(batch):
    """推理/验证：stretch 为原图尺寸；letterbox 为先 pad 到 INPUT_SIZE 再交给 processor。"""
    images = []
    target_sizes = []
    image_ids = []
    letterbox_meta: list[dict | None] = []
    for img, image_id, _ in batch:
        if img.mode != "RGB":
            img = img.convert("RGB")
        orig_w, orig_h = img.size
        if uses_letterbox():
            img, _ = letterbox_pil_coco(img, [])
            ratio, _, _, pad_w, pad_h = letterbox_params(orig_w, orig_h, pc.INPUT_SIZE)
            letterbox_meta.append(
                {
                    "ratio": ratio,
                    "pad_w": pad_w,
                    "pad_h": pad_h,
                    "orig_w": orig_w,
                    "orig_h": orig_h,
                }
            )
            target_sizes.append((pc.INPUT_SIZE, pc.INPUT_SIZE))
        else:
            letterbox_meta.append(None)
            target_sizes.append((orig_h, orig_w))
        images.append(img)
        image_ids.append(image_id)
    return images, torch.tensor(target_sizes, dtype=torch.int64), image_ids, letterbox_meta


def _xyxy_to_xywh(box: torch.Tensor) -> list[float]:
    x1, y1, x2, y2 = box.tolist()
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


@torch.no_grad()
def evaluate_coco_bbox_map(
    model: Deimv2ForObjectDetection,
    processor,
    dataset: TeaCocoDataset,
    coco_gt: COCO,
    device: torch.device,
    batch_size: int,
    score_threshold: float,
    num_workers: int,
    *,
    label_to_coco_id: dict[int, int] | None = None,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_infer_only,
        pin_memory=device.type == "cuda",
    )
    predictions: list[dict] = []
    for images, target_sizes_cpu, image_ids, lb_meta in loader:
        enc = processor(images=list(images), return_tensors="pt")
        pixel_values = enc["pixel_values"].to(device)
        target_sizes = target_sizes_cpu.to(device)
        kwargs = {"pixel_values": pixel_values}
        if "pixel_mask" in enc:
            kwargs["pixel_mask"] = enc["pixel_mask"].to(device)
        outputs = model(**kwargs)
        results = processor.post_process_object_detection(
            outputs, threshold=score_threshold, target_sizes=target_sizes
        )
        for img_id, res, meta in zip(image_ids, results, lb_meta):
            scores = res["scores"]
            labels = res["labels"]
            boxes = res["boxes"]
            for s, lab, box in zip(scores.tolist(), labels.tolist(), boxes):
                lab_i = int(lab)
                if label_to_coco_id is not None:
                    if lab_i not in label_to_coco_id:
                        raise ValueError(
                            f"模型输出 label={lab_i} 不在 [0, {len(label_to_coco_id)}) 内"
                        )
                    coco_cid = label_to_coco_id[lab_i]
                else:
                    coco_cid = lab_i
                if meta is not None:
                    xyxy = letterbox_inverse_xyxy(
                        box.cpu().numpy(),
                        ratio=meta["ratio"],
                        pad_w=meta["pad_w"],
                        pad_h=meta["pad_h"],
                        orig_w=meta["orig_w"],
                        orig_h=meta["orig_h"],
                    )
                    x1, y1, x2, y2 = (float(v) for v in xyxy)
                    bbox_xywh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
                else:
                    bbox_xywh = _xyxy_to_xywh(box.cpu())
                predictions.append(
                    {
                        "image_id": int(img_id),
                        "category_id": coco_cid,
                        "bbox": bbox_xywh,
                        "score": float(s),
                    }
                )

    if not predictions:
        first_id = int(coco_gt.getImgIds()[0])
        dummy_cid = (
            next(iter(label_to_coco_id.values()))
            if label_to_coco_id
            else 0
        )
        predictions.append(
            {
                "image_id": first_id,
                "category_id": dummy_cid,
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "score": 1e-5,
            }
        )

    buf_out, buf_err = StringIO(), StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        coco_dt = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    s = coco_eval.stats
    return {
        "bbox_mAP": float(s[0]),
        "bbox_mAP_50": float(s[1]),
        "bbox_mAP_75": float(s[2]),
        "bbox_mAR_100": float(s[8]),
    }


_MAP_METRIC_KEYS = ("bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "bbox_mAR_100")


def _mean_map_metrics(per_dataset: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_dataset:
        return {k: 0.0 for k in _MAP_METRIC_KEYS}
    n = len(per_dataset)
    return {k: sum(m[k] for m in per_dataset.values()) / n for k in _MAP_METRIC_KEYS}


@torch.no_grad()
def evaluate_val_maps_per_dataset(
    model: Deimv2ForObjectDetection,
    processor,
    val_sources: list[ValEvalSource],
    device: torch.device,
    batch_size: int,
    score_threshold: float,
    num_workers: int,
    *,
    label_to_coco_id: dict[int, int] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """对各数据集 val 分别评测，逐行打印，返回 (per_dataset, mean)。"""
    per_ds: dict[str, dict[str, float]] = {}
    for src in val_sources:
        buf = StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            coco_gt = COCO(str(src.ann_path))
        m = evaluate_coco_bbox_map(
            model,
            processor,
            src.dataset,
            coco_gt,
            device,
            batch_size=batch_size,
            score_threshold=score_threshold,
            num_workers=num_workers,
            label_to_coco_id=label_to_coco_id,
        )
        per_ds[src.name] = m
        print(
            f"  val [{src.name}] mAP={m['bbox_mAP']:.4f} mAP50={m['bbox_mAP_50']:.4f} "
            f"mAP75={m['bbox_mAP_75']:.4f} AR100={m['bbox_mAR_100']:.4f}"
        )
    mean_m = _mean_map_metrics(per_ds)
    if len(per_ds) > 1:
        print(
            f"  val [mean] mAP={mean_m['bbox_mAP']:.4f} mAP50={mean_m['bbox_mAP_50']:.4f} "
            f"mAP75={mean_m['bbox_mAP_75']:.4f} AR100={mean_m['bbox_mAR_100']:.4f}"
        )
    return per_ds, mean_m


def apply_train_mode(
    model: Deimv2ForObjectDetection, train_mode: str, loss_bbox_scale: float
) -> None:
    if train_mode == "backbone_frozen":
        freeze_backbone_only(model)
        restore_default_deimv2_loss_weights(model, loss_bbox_scale)
    elif train_mode == "heads_only":
        freeze_all_except_detection_heads(model)
        restore_default_deimv2_loss_weights(model, loss_bbox_scale)
    elif train_mode == "classification_only":
        freeze_all_except_classification_heads(model)
        set_classification_only_loss(model)
    else:
        raise ValueError(f"未知 train_mode: {train_mode}")


def _sum_weighted_cls_loss(loss_dict: dict | None) -> float:
    """从 HF Deimv2 loss_dict 汇总分类侧加权项（MAL / CE / focal / VFL 等，含 aux / dn 后缀）。"""
    if not loss_dict:
        return 0.0
    s = 0.0
    for k, v in loss_dict.items():
        if k in ("loss_ce", "loss_cls") or "loss_mal" in k or "loss_focal" in k or "loss_vfl" in k:
            s += float(v.detach())
    return s


def _sum_weighted_l1_bbox_loss(loss_dict: dict | None) -> float:
    """汇总 L1 框损失加权项（含 loss_bbox_aux_*、loss_bbox_dn_* 等）。"""
    if not loss_dict:
        return 0.0
    s = 0.0
    for k, v in loss_dict.items():
        if k.startswith("loss_bbox"):
            s += float(v.detach())
    return s


def _sum_weighted_giou_loss(loss_dict: dict | None) -> float:
    """汇总 GIoU 损失加权项（含 loss_giou_aux_*、loss_giou_dn_* 等）。"""
    if not loss_dict:
        return 0.0
    s = 0.0
    for k, v in loss_dict.items():
        if k.startswith("loss_giou"):
            s += float(v.detach())
    return s


# ========================================================================
# Train Checkpoint
# ========================================================================

def _load_training_state_path(resume_dir: Path) -> Path | None:
    p = resume_dir / "training_state.pt"
    return p if p.is_file() else None


def _infer_completed_epoch(resume_dir: Path) -> int | None:
    """从 train_metrics.json 或目录名 checkpoint-epochN 推断已完成的 epoch。"""
    metrics_path = resume_dir / "train_metrics.json"
    if metrics_path.is_file():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                return int(json.load(f)["epoch"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    name = resume_dir.name
    prefix = "checkpoint-epoch"
    if name.startswith(prefix):
        try:
            return int(name[len(prefix) :])
        except ValueError:
            pass
    return None


_CHECKPOINT_ARTIFACTS = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "train_metrics.json",
    "training_state.pt",
)


def _val_bbox_map(map_val: dict[str, float] | None) -> float | None:
    if not map_val:
        return None
    v = map_val.get("bbox_mAP")
    return float(v) if v is not None else None


def _load_val_bbox_map_from_metrics(metrics_path: Path) -> tuple[float | None, int | None]:
    if not metrics_path.is_file():
        return None, None
    try:
        with open(metrics_path, encoding="utf-8") as f:
            data = json.load(f)
        return _val_bbox_map(data.get("val_map")), int(data["epoch"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, None


def _load_best_val_bbox_map(output_dir: Path) -> tuple[float | None, int | None]:
    """从 checkpoint-best/train_metrics.json 读取历史最佳 val bbox mAP。"""
    return _load_val_bbox_map_from_metrics(output_dir / "checkpoint-best" / "train_metrics.json")


def _promote_checkpoint_dir(src: Path, dst: Path) -> None:
    """将一轮 checkpoint 目录中的标准产物复制到 final / checkpoint-best 等目标目录。"""
    dst.mkdir(parents=True, exist_ok=True)
    for name in _CHECKPOINT_ARTIFACTS:
        s = src / name
        if s.is_file():
            shutil.copy2(s, dst / name)


def _build_training_state_payload(
    *,
    epoch: int,
    opt: AdamW,
    model: Deimv2ForObjectDetection,
    unfreeze_backbone_last_n: int,
    warmup_epochs: int,
    pretrained: str,
    load_source: str,
) -> dict:
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return {
        "epoch": epoch,
        "optimizer": opt.state_dict(),
        "optimizer_named_state": pack_adamw_named_state(opt, model),
        "rng_cpu": torch.random.get_rng_state(),
        "rng_cuda": rng_cuda,
        "unfreeze_backbone_last_n": unfreeze_backbone_last_n,
        "warmup_epochs": warmup_epochs,
        "pretrained": pretrained,
        "load_source": load_source,
    }


def _save_epoch_checkpoint(
    save_dir: Path,
    *,
    model: Deimv2ForObjectDetection,
    processor,
    metrics: dict,
    opt: AdamW,
    epoch: int,
    unfreeze_backbone_last_n: int,
    warmup_epochs: int,
    pretrained: str,
    load_source: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    with open(save_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    torch.save(
        _build_training_state_payload(
            epoch=epoch,
            opt=opt,
            model=model,
            unfreeze_backbone_last_n=unfreeze_backbone_last_n,
            warmup_epochs=warmup_epochs,
            pretrained=pretrained,
            load_source=load_source,
        ),
        save_dir / "training_state.pt",
    )


# ========================================================================
# Train Data
# ========================================================================

@dataclass
class ValEvalSource:
    """单个数据集的 val 评测源（每轮单独算 mAP）。"""

    name: str
    root: Path
    dataset: TeaCocoDataset
    ann_path: Path


def build_train_val_sources(
    args,
    *,
    script_dir: Path,
    category_id_remap: dict[int, int] | None = None,
) -> tuple[TeaCocoDataset, Path, list[str], Path | None, list[ValEvalSource], list[float] | None]:
    """
    返回 (train_ds, train_ann_path, roots_used, merged_cache_dir_or_None, val_eval_sources, train_weights)。
    train：单集用该集 train；多集用合并后的 merged_train.json。
    val 评测：每个根目录各一份原生 val（不合并）。
    """
    train_mosaic_p = resolve_train_mosaic_p(args.aug_level, mosaic_p=args.aug_det_mosaic_p)
    roots: list[Path] = list(args.datasets)

    val_sources: list[ValEvalSource] = []
    for root in roots:
        _, val_ann = _default_coco_train_val_paths(root)
        val_sources.append(
            ValEvalSource(
                name=root.name,
                root=root,
                dataset=TeaCocoDataset(
                    root, val_ann, mosaic_p=0.0, category_id_remap=category_id_remap
                ),
                ann_path=val_ann,
            )
        )

    if len(roots) == 1:
        if args.dataset_ratios:
            print("提示: 仅 1 个 --datasets 时忽略 --dataset_ratios。")
        train_ann, _ = _default_coco_train_val_paths(roots[0])
        train_ds = TeaCocoDataset(
            roots[0], train_ann, mosaic_p=train_mosaic_p, category_id_remap=category_id_remap
        )
        return train_ds, train_ann, [str(roots[0])], None, val_sources, None

    fingerprint = hashlib.md5("\n".join(str(r.resolve()) for r in roots).encode("utf-8")).hexdigest()[:16]
    merged_dir = args.output_dir / "merged_coco_cache" / fingerprint
    train_ann_m, _val_ann_m, train_weights = merge_coco_roots_for_training(
        roots, merged_dir, dataset_ratios=args.dataset_ratios
    )
    train_ds = TeaCocoDataset(
        script_dir,
        train_ann_m,
        mosaic_p=train_mosaic_p,
        sample_weights=train_weights,
        category_id_remap=category_id_remap,
    )
    return train_ds, train_ann_m, [str(r) for r in roots], merged_dir, val_sources, train_weights


# ========================================================================
# Train Curves
# ========================================================================

import re

from utils.common import setup_chinese_font

_CHECKPOINT_RE = re.compile(r"^checkpoint-epoch(\d+)$")
_MARKER_EPOCH_LIMIT = 100
MAP_COCO_KEY = "bbox_mAP"
MAP50_KEY = "bbox_mAP_50"
_MEAN_DASH = (0, (3, 1.5))
MAP_SUBPLOT_TITLE = "检测 mAP（IoU 0.50:0.95）"
MAP50_SUBPLOT_TITLE = "检测 mAP@0.5（IoU 0.50）"
MAP_YLABEL = "mAP"
_LINE_COLORS = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#CC78BC",
    "#D55E00",
    "#56B4E9",
    "#CA9161",
    "#949494",
    "#ECE133",
    "#FBAFE4",
    "#00BFC4",
    "#7A5195",
]
_LINE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "p", "*", "<", ">"]

_CHINESE_FONT = setup_chinese_font()


def resolve_run_dirs(
    positional: list[Path],
    legacy: list[Path] | None,
    outputs_root: Path,
) -> list[Path]:
    raw = list(positional)
    if legacy:
        raw.extend(legacy)
    if not raw:
        raw = [outputs_root / "deimv2_s_tealeaves"]
    resolved: list[Path] = []
    root = outputs_root.expanduser().resolve()
    for p in raw:
        p = p.expanduser()
        if p.is_absolute() or p.exists():
            resolved.append(p.resolve())
            continue
        resolved.append((root / p).resolve())
    return resolved


def resolve_curves_dir(outputs_root: Path, curves_dir: Path | None) -> Path:
    if curves_dir is not None:
        return curves_dir.expanduser().resolve()
    return (outputs_root.expanduser().resolve() / "train_curves").resolve()


def map_metric(m: dict | None, key: str, fallback: float = 0.0) -> float:
    if not m:
        return fallback
    return float(m.get(key, fallback))


def _path_basename(p: object) -> str:
    if p is None:
        return "—"
    s = str(p).strip()
    if not s or s.lower() == "none":
        return "—"
    return Path(s).name


def _infer_val_name_from_run_dir(run_dir: Path) -> str | None:
    """无 dataset_roots 的旧 checkpoint：从输出目录名推断（如 deimv2_l_march）。"""
    n = run_dir.name.lower()
    if "april_iy" in n or "april-iy" in n:
        return "teabud_april_IY"
    if "march" in n and "april" in n:
        return None
    if "march" in n:
        return "teabud_march_ztu"
    if "april" in n:
        return "teabud_april"
    return None


def _infer_single_val_dataset_name(metrics: dict, run_dir: Path | None = None) -> str | None:
    """旧版 metrics 无 val_map_per_dataset 时，从 dataset_roots 或 run 目录名推断单集 val 名称。"""
    roots = metrics.get("dataset_roots")
    if isinstance(roots, list) and len(roots) == 1:
        name = _path_basename(roots[0])
        if name != "—":
            return name
    per_ds = metrics.get("val_map_per_dataset")
    if isinstance(per_ds, dict) and len(per_ds) == 1:
        only = next(iter(per_ds))
        if str(only) != "val":
            return str(only)
    if run_dir is not None:
        return _infer_val_name_from_run_dir(run_dir)
    return None


def _resolve_val_dataset_keys(series: RunSeries, inferred_name: str | None = None) -> None:
    """将占位键 val 替换为真实数据集目录名（单 val 集、旧 metrics 格式）。"""
    if len(series.map_coco.val_per_dataset) != 1 or "val" not in series.map_coco.val_per_dataset:
        return
    name = (
        inferred_name
        or _infer_single_val_dataset_name(series.config, series.run_dir)
        or _infer_val_name_from_run_dir(series.run_dir)
    )
    if not name:
        return
    for tracks in (series.map_coco, series.map50):
        if "val" in tracks.val_per_dataset:
            tracks.val_per_dataset[name] = tracks.val_per_dataset.pop("val")


def _val_legend_label(ds_name: str) -> str:
    return f"val · {_path_basename(ds_name)}"


def _fmt_float(x: object, sci_threshold: float = 1e-3) -> str:
    if x is None:
        return "—"
    v = float(x)
    if abs(v) < sci_threshold or abs(v) >= 1e4:
        return f"{v:.2e}"
    return f"{v:g}"


def format_training_config(metrics: dict, run_name: str, epoch_first: int, epoch_last: int) -> str:
    opt = metrics.get("optimizer") or {}
    aug = metrics.get("augmentation") or {}
    loss_w = metrics.get("loss_weights") or {}

    pretrained = metrics.get("pretrained") or "—"

    lr = opt.get("lr")
    lr_bb = opt.get("lr_backbone")
    lr_part = f"lr={_fmt_float(lr)}"
    if opt.get("layerwise_dinov3_backbone_lr") and lr_bb is not None:
        lr_part += f", lr_backbone={_fmt_float(lr_bb)}, decay={opt.get('backbone_lr_decay', '—')}"
    lr_part += f", wd={_fmt_float(opt.get('weight_decay'))}"

    roots = metrics.get("dataset_roots")
    if isinstance(roots, list) and roots:
        data_part = " + ".join(_path_basename(r) for r in roots)
    else:
        data_part = "（见 val 曲线图例 / 未记录 dataset_roots）"

    ratios = metrics.get("dataset_ratios")
    if isinstance(ratios, list) and ratios:
        data_part += f"  ratios={ratios}"
    if metrics.get("train_sample_weights_enabled"):
        data_part += "  [sample_weights]"

    aug_lines = format_augmentation_summary(aug)

    return "\n".join(
        [
            f"训练目录: {run_name}    epoch {epoch_first}–{epoch_last}",
            (
                f"骨架: {pretrained}    模式: {metrics.get('train_mode', '—')}    "
                f"解冻 backbone 末 {metrics.get('unfreeze_backbone_last_n', '—')} 层    "
                f"warmup_epochs={metrics.get('warmup_epochs', 0)}"
            ),
            f"优化器: {lr_part}    param_groups={opt.get('param_groups', '—')}",
            f"训练数据: {data_part}",
            *aug_lines,
            (
                f"mAP score_thr={metrics.get('map_score_threshold', '—')}    "
                f"初始模型权重: {_path_basename(metrics.get('load_source'))}"
            ),
            (
                f"loss 权重: mal={loss_w.get('mal', '—')}, bbox={loss_w.get('bbox', '—')}, "
                f"giou={loss_w.get('giou', '—')}, fgl={loss_w.get('fgl', '—')}, ddf={loss_w.get('ddf', '—')}"
            ),
        ]
    )


@dataclass
class MapTracks:
    """一组 mAP 曲线（train / 各 val / mean）。"""

    train: list[float] = field(default_factory=list)
    val_per_dataset: dict[str, list[float]] = field(default_factory=dict)
    val_mean: list[float] | None = None


@dataclass
class RunSeries:
    run_dir: Path
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    map_coco: MapTracks = field(default_factory=MapTracks)
    map50: MapTracks = field(default_factory=MapTracks)
    config: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.run_dir.name


def iter_checkpoint_metrics(run_dir: Path, max_epoch: int | None) -> list[tuple[int, dict]]:
    rows: list[tuple[int, dict]] = []
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        m = _CHECKPOINT_RE.match(child.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if max_epoch is not None and epoch > max_epoch:
            continue
        metrics_path = child / "train_metrics.json"
        if not metrics_path.is_file():
            continue
        with open(metrics_path, encoding="utf-8-sig") as f:
            rows.append((epoch, json.load(f)))
    rows.sort(key=lambda x: x[0])
    return rows


def _default_val_dataset_key(run_dir: Path, metrics: dict | None = None) -> str:
    """无 dataset_roots 时的回退键（单数据集 run 目录名推断）。"""
    if metrics is not None:
        name = _infer_single_val_dataset_name(metrics, run_dir)
        if name:
            return name
    return _infer_val_name_from_run_dir(run_dir) or "val"


def _val_dataset_key_from_metrics(metrics: dict, fallback: str) -> str:
    """按当前 epoch 的 metrics 判断 val 所属数据集（支持训练中途切换 val 集）。"""
    per_ds = metrics.get("val_map_per_dataset")
    if isinstance(per_ds, dict) and per_ds:
        if len(per_ds) == 1:
            only = next(iter(per_ds))
            if str(only) != "val":
                return str(only)
        return fallback
    roots = metrics.get("dataset_roots")
    if isinstance(roots, list) and len(roots) == 1:
        name = _path_basename(roots[0])
        if name != "—":
            return name
    return fallback


def _record_val_map_points(
    val_points: dict[str, dict[int, float]],
    epoch: int,
    metrics: dict,
    map_key: str,
    default_key: str,
) -> None:
    per_ds = metrics.get("val_map_per_dataset")
    if isinstance(per_ds, dict) and per_ds:
        for name, m in per_ds.items():
            if isinstance(m, dict):
                val_points.setdefault(str(name), {})[epoch] = map_metric(m, map_key)
    else:
        vm = metrics.get("val_map") or {}
        key = _val_dataset_key_from_metrics(metrics, default_key)
        val_points.setdefault(key, {})[epoch] = map_metric(vm, map_key)


def _merge_val_alias(val_points: dict[str, dict[int, float]], alias: str, canonical: str) -> None:
    if alias not in val_points or alias == canonical:
        return
    target = val_points.setdefault(canonical, {})
    for ep, value in val_points.pop(alias).items():
        target.setdefault(ep, value)


def _consolidate_val_points(val_points: dict[str, dict[int, float]], run_dir: Path) -> None:
    """仅合并同一数据集、仅 metrics 格式不同的 val 占位键（epoch 有重叠时才合并）。"""
    if "val" not in val_points:
        return
    val_epochs = set(val_points["val"])
    for name in list(val_points):
        if name == "val":
            continue
        if val_epochs & set(val_points[name]):
            _merge_val_alias(val_points, "val", name)
            return
    if len(val_points) == 1:
        canonical = _infer_val_name_from_run_dir(run_dir)
        if canonical and canonical != "val":
            val_points[canonical] = val_points.pop("val")


def _val_lists_from_points(
    val_points: dict[str, dict[int, float]],
    epochs: list[int],
) -> dict[str, list[float]]:
    return {name: [ep_map.get(ep, float("nan")) for ep in epochs] for name, ep_map in val_points.items()}


def load_series(run_dir: Path, max_epoch: int | None) -> RunSeries:
    series = RunSeries(run_dir=run_dir.resolve())
    mean_coco: list[float] = []
    mean_50: list[float] = []
    track_mean = False
    last_metrics: dict = {}
    inferred_val_name: str | None = None
    val_coco_points: dict[str, dict[int, float]] = {}
    val50_points: dict[str, dict[int, float]] = {}
    default_val_key = _default_val_dataset_key(series.run_dir)

    for epoch, d in iter_checkpoint_metrics(run_dir, max_epoch):
        last_metrics = d
        if inferred_val_name is None:
            inferred_val_name = _infer_single_val_dataset_name(d, series.run_dir)
        ep = int(d.get("epoch", epoch))
        val_key = _val_dataset_key_from_metrics(d, default_val_key)
        series.epochs.append(ep)
        series.train_loss.append(float(d["train_loss"]))

        tm = d.get("train_map") or {}
        series.map_coco.train.append(map_metric(tm, MAP_COCO_KEY, fallback=map_metric(d, MAP_COCO_KEY)))
        series.map50.train.append(map_metric(tm, MAP50_KEY))

        _record_val_map_points(val_coco_points, ep, d, MAP_COCO_KEY, val_key)
        _record_val_map_points(val50_points, ep, d, MAP50_KEY, val_key)

        mean_m = d.get("val_map_mean")
        if isinstance(mean_m, dict) and mean_m:
            track_mean = True
            mean_coco.append(map_metric(mean_m, MAP_COCO_KEY))
            mean_50.append(map_metric(mean_m, MAP50_KEY))
        else:
            mean_coco.append(float("nan"))
            mean_50.append(float("nan"))

    _consolidate_val_points(val_coco_points, series.run_dir)
    _consolidate_val_points(val50_points, series.run_dir)
    series.map_coco.val_per_dataset = _val_lists_from_points(val_coco_points, series.epochs)
    series.map50.val_per_dataset = _val_lists_from_points(val50_points, series.epochs)

    if track_mean:
        series.map_coco.val_mean = mean_coco
        series.map50.val_mean = mean_50
    series.config = last_metrics
    _resolve_val_dataset_keys(series, inferred_val_name)
    return series


def _line_style(index: int, use_markers: bool) -> dict:
    style: dict = {
        "color": _LINE_COLORS[index % len(_LINE_COLORS)],
        "linewidth": 1.8,
    }
    if use_markers:
        style["marker"] = _LINE_MARKERS[index % len(_LINE_MARKERS)]
        style["markersize"] = 4
    return style


def _plot_line(
    ax,
    epochs: list[int],
    y: list[float],
    index: int,
    label: str,
    use_markers: bool,
    *,
    dense_dash: bool = False,
) -> None:
    kw = _line_style(index, use_markers)
    if dense_dash:
        kw["linestyle"] = _MEAN_DASH
    ax.plot(epochs, y, label=label, **kw)


def _plot_sparse_line(
    ax,
    epochs: list[int],
    y: list[float],
    index: int,
    label: str,
    use_markers: bool,
    *,
    dense_dash: bool = False,
) -> None:
    """仅绘制有值的 epoch（训练中途切换 val 集时各数据集各画一段）。"""
    xs = [e for e, v in zip(epochs, y) if v == v]
    ys = [v for v in y if v == v]
    if not xs:
        return
    kw = _line_style(index, use_markers)
    if dense_dash:
        kw["linestyle"] = _MEAN_DASH
    ax.plot(xs, ys, label=label, **kw)


def _plot_map_subplot(
    ax,
    epochs: list[int],
    tracks: MapTracks,
    title: str,
    use_markers: bool,
) -> None:
    _plot_line(ax, epochs, tracks.train, 0, "train", use_markers)
    color_idx = 1
    for ds_name, values in tracks.val_per_dataset.items():
        if len(values) == len(epochs) and any(v == v for v in values):
            _plot_sparse_line(
                ax, epochs, values, color_idx, _val_legend_label(ds_name), use_markers
            )
            color_idx += 1
    if (
        tracks.val_mean is not None
        and len(tracks.val_mean) == len(epochs)
        and any(v == v for v in tracks.val_mean)
    ):
        _plot_sparse_line(
            ax, epochs, tracks.val_mean, color_idx, "val · 均值", use_markers, dense_dash=True
        )
    ax.set_ylabel(MAP_YLABEL)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7.5)
    ax.grid(True, alpha=0.3)


def plot_run(series: RunSeries, out_path: Path) -> None:
    epochs = series.epochs
    if not epochs:
        raise ValueError(f"未找到 train_metrics.json: {series.run_dir}")

    use_markers = len(epochs) <= _MARKER_EPOCH_LIMIT
    config_text = format_training_config(series.config, series.name, epochs[0], epochs[-1])

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(f"训练曲线 — {series.name}", fontsize=12, y=0.98)

    _plot_line(axes[0], epochs, series.train_loss, 0, "train loss", use_markers)
    axes[0].set_ylabel("train_loss")
    axes[0].set_title("Loss")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    _plot_map_subplot(axes[1], epochs, series.map_coco, MAP_SUBPLOT_TITLE, use_markers)
    _plot_map_subplot(axes[2], epochs, series.map50, MAP50_SUBPLOT_TITLE, use_markers)
    axes[2].set_xlabel("epoch")

    fig.text(
        0.5,
        0.01,
        config_text,
        ha="center",
        va="bottom",
        fontsize=7.5,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f7f7", edgecolor="#cccccc", alpha=0.95),
    )

    fig.subplots_adjust(top=0.94, bottom=0.18, hspace=0.38)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def default_out_path(curves_dir: Path, run_name: str, last_epoch: int) -> Path:
    return curves_dir / f"{run_name}_epoch{last_epoch}.png"


_CHINESE_FONT = setup_chinese_font()