"""
使用 DEIMv2（DINOv3 骨干）在茶叶 COCO 数据集上做迁移学习。

通过 --pretrained / --pretrained_id 从 HuggingFace Hub 拉取 Transformers 版 Deimv2 权重（默认 DINOv3-S；
可选 DINOv3-L 等大骨干）。Intellindust/DEIMv2_DINOv3_L_COCO 等为原版仓库配置，不能用于本脚本的 from_pretrained。

使用 HuggingFace Transformers 中的 Deimv2ForObjectDetection / AutoImageProcessor，便于训练与保存。

默认（--train_mode backbone_frozen）：只冻结 conv_encoder 内的 backbone（DINOv3 为
model.conv_encoder.backbone.*，CNN 骨干为 model.conv_encoder.model.*）；neck（如 STA、
fusion_proj）、HybridEncoder、Decoder、分类与框回归等其余参数全部训练，并使用完整
检测损失。可用 --loss_bbox_scale 放大 bbox/giou 权重。
可选 --unfreeze_backbone_last_n：在各类 train_mode 的冻结策略之后，对 DINOv3 ViT 骨干再解冻最后若干 Transformer block；
当 n>0 且骨干为 dinov3_vit 时，自动对「可训练的」conv_encoder.backbone.* 使用按 block 分层学习率（lr_backbone、backbone_lr_decay），
与 deimv2_dinov3_l_coco.yml 思路一致：该部分 norm/bias 使用 weight_decay=0，其余权重使用 --weight_decay；检测头/neck 等仍用 --lr。
可选 --warmup_epochs：前若干个 epoch 对骨干参数组（conv_encoder.backbone / conv_encoder.model）线性 lr 热身（非骨干组不使用）。

其它模式：--train_mode heads_only 仅训练各检测头（仍冻 encoder/decoder 主体）；
--train_mode classification_only 为仅分类头且 bbox 类损失置 0（易 mAP 接近 0）。

断点续训：使用 --resume_from 指向某次保存目录（如 checkpoint-epoch5）。若该目录含
training_state.pt（每轮保存时会写入），将恢复优化器与 RNG；否则仅加载模型权重，
并根据 train_metrics.json 或目录名推断下一 epoch。

依赖：torch, torchvision, transformers, scipy（Hungarian 匹配需要）, pycocotools（验证 mAP）

训练数据增强在 collate 中完成，由 --aug_preset 选择：
  detection（默认）：RandomPhotometricDistort、RandomZoomOut、RandomIoUCrop（默认中等偏强概率）、
  SanitizeBoundingBoxes、RandomHorizontalFlip、Resize(640)；训练集可选四图 Mosaic（--aug_det_mosaic_p）。
  torchvision 无标准「检测框正确」的 CutMix；Mosaic 通过拼接四张图与框偏移丰富场景。
  simple：仅水平翻转 + ColorJitter；none：不增强。验证/mAP 始终无增强。

多数据集：--datasets 可传一个或多个 COCO 根目录。单个时直接训练；多个时合并 train 用于训练，
各目录 val 在每轮单独算 mAP 并打印，最后输出各指标平均值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

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
from transformers import AutoImageProcessor, Deimv2ForObjectDetection
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

# 预设须为 Transformers 兼容仓库（含 Deimv2Config / preprocessor）。L 档使用社区转换权重。
HF_DEIMV2_PRESETS: dict[str, str] = {
    "dinov3_s": "harshaljanjani/DEIMv2_DINOv3_S_COCO_Transformers",
    "dinov3_l": "alessioarcara/deimv2-deimv2_dinov3_l_coco",
}


def resolve_pretrained_hub_id(pretrained: str, pretrained_id: str | None) -> str:
    """返回 HuggingFace Hub repo id；--pretrained_id 非空时优先于 --pretrained 预设。"""
    if pretrained_id is not None and str(pretrained_id).strip():
        return str(pretrained_id).strip()
    if pretrained not in HF_DEIMV2_PRESETS:
        raise ValueError(f"未知 --pretrained={pretrained!r}，可选: {tuple(HF_DEIMV2_PRESETS)}")
    return HF_DEIMV2_PRESETS[pretrained]


_SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets",
        nargs="+",
        type=Path,
        default=[Path(r"./datasets/TeaLeavesDatasets_split_lr")],
        metavar="DIR",
        help="一个或多个 COCO 数据集根目录（含 images/ 与 annotations/）。"
        "传 1 个则单集训练；传多个则合并 train 训练，每轮对各集 val 分别评测并打印均值。"
        "相对路径相对于本脚本所在目录。",
    )
    p.add_argument(
        "--dataset_ratios",
        nargs="*",
        type=float,
        default=None,
        metavar="RATIO",
        help="与 --datasets 一一对应的训练采样比例（仅多个数据集时生效）。"
        "例如 1 2 表示两集被抽到的概率比为 1:2，与各自图片张数无关。"
        "不写则按各集 train 图片数量自然配比（与合并后均匀 shuffle 一致）。",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path(r"./outputs/deimv2_s_tealeaves"),
        help="checkpoint 与日志输出目录。相对路径相对于本脚本所在目录",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--pretrained",
        type=str,
        choices=tuple(HF_DEIMV2_PRESETS),
        default="dinov3_s",
        help="HF Hub 上的 DEIMv2+DINOv3 预训练预设（未指定 --pretrained_id 时生效）。dinov3_l 对应社区 Transformers 版大骨干权重。",
    )
    p.add_argument(
        "--pretrained_id",
        type=str,
        default=None,
        help="覆盖 --pretrained：任意 Hub 上的 Deimv2ForObjectDetection 兼容仓库 id（须含 config 与 preprocessor）。",
    )
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--train_mode",
        type=str,
        choices=("backbone_frozen", "heads_only", "classification_only"),
        default="backbone_frozen",
        help="backbone_frozen=只冻骨干其余全训；heads_only=只训检测头；classification_only=仅分类且框损失为 0",
    )
    p.add_argument(
        "--loss_bbox_scale",
        type=float,
        default=1.0,
        help="在 backbone_frozen / heads_only 下，将 bbox 与 giou 损失权重相对默认再乘该系数",
    )
    p.add_argument(
        "--unfreeze_backbone_last_n",
        type=int,
        default=0,
        help="仅骨干为 DINOv3 ViT 时生效：在 train_mode 冻结策略之后，再解冻 backbone 最后 n 个 Transformer block（及 conv_encoder.backbone.norm）。"
        "n>0 时自动仅对可训练的 conv_encoder.backbone.* 使用 lr_backbone / backbone_lr_decay 分层与 norm/bias 的 weight_decay=0；"
        "检测头/neck 等仍用 --lr。0=关闭。可大于总层数，将按 num_hidden_layers 封顶。",
    )
    p.add_argument(
        "--lr_backbone",
        type=float,
        default=None,
        help="与 --unfreeze_backbone_last_n>0 且 DINOv3 骨干联用：最后一层 ViT block 与 conv_encoder.backbone.norm 的学习率；默认 0.125×--lr",
    )
    p.add_argument(
        "--backbone_lr_decay",
        type=float,
        default=1.0,
        help="与分层骨干联用：由深到浅每浅一层多乘该因子；1.0 表示各解冻 block 均为 lr_backbone",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="AdamW 的 weight_decay；在启用骨干分层时，对 conv_encoder.backbone 中 norm/bias 等自动置 0，其余权重与检测部分仍用该值",
    )
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=0,
        help="前若干个 epoch 仅对骨干参数组（conv_encoder.backbone / conv_encoder.model）做线性 lr 热身："
        "mult=(epoch-1)/(W-1) 自 0→1，lr=base_lr×mult；非骨干组恒为 base_lr。0=关闭。",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--map_score_threshold",
        type=float,
        default=0.05,
        help="验证 mAP 时过滤低分框的阈值（与训练无关）",
    )
    p.add_argument(
        "--map_batch_size",
        type=int,
        default=None,
        help="算 mAP 时的 batch，默认与 --batch_size 相同",
    )
    p.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="从某次保存的目录恢复（含 config.json、模型权重）。若同目录有 training_state.pt 则一并恢复优化器并从下一 epoch 继续；相对路径相对于本脚本所在目录",
    )
    p.add_argument(
        "--aug_preset",
        type=str,
        choices=("detection", "simple", "none"),
        default="detection",
        help="detection=torchvision v2 光度+ZoomOut+IoUCrop+翻转+Resize640（默认较难但可学）；"
        "训练集可叠加四图 Mosaic（见 --aug_det_mosaic_p）。"
        "simple=仅翻转+ColorJitter；none=不增强",
    )
    p.add_argument(
        "--aug_simple_flip_p",
        type=float,
        default=0.5,
        help="aug_preset=simple 时水平翻转概率",
    )
    p.add_argument(
        "--aug_simple_color_p",
        type=float,
        default=0.8,
        help="aug_preset=simple 时 ColorJitter 应用概率",
    )
    p.add_argument(
        "--aug_det_photometric_p",
        type=float,
        default=0.7,
        help="aug_preset=detection：RandomPhotometricDistort 概率",
    )
    p.add_argument(
        "--aug_det_zoomout_fill",
        type=float,
        default=0.0,
        help="aug_preset=detection：RandomZoomOut 的 fill",
    )
    p.add_argument(
        "--aug_det_zoomout_p",
        type=float,
        default=0.6,
        help="aug_preset=detection：RandomZoomOut 概率",
    )
    p.add_argument(
        "--aug_det_iou_crop_p",
        type=float,
        default=0.85,
        help="aug_preset=detection：RandomIoUCrop 以 RandomApply 包裹的应用概率",
    )
    p.add_argument(
        "--aug_det_flip_p",
        type=float,
        default=0.5,
        help="aug_preset=detection：RandomHorizontalFlip 概率",
    )
    p.add_argument(
        "--aug_det_mosaic_p",
        type=float,
        default=0.3,
        help="仅 aug_preset=detection 且训练集 len>=4：以该概率在 __getitem__ 内做四宫格 Mosaic（CutMix 需专门框逻辑，torchvision 未内置检测 CutMix）。0=关闭",
    )
    return p.parse_args()

def _resolve_cli_path(path: Path) -> Path:
    """相对路径相对于本脚本所在目录解析（不依赖 shell 当前工作目录）。绝对路径仍按系统规则 resolve。"""
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_SCRIPT_DIR / path).resolve()


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


@dataclass
class ValEvalSource:
    """单个数据集的 val 评测源（每轮单独算 mAP）。"""

    name: str
    root: Path
    dataset: TeaCocoDataset
    ann_path: Path


def _build_train_val_sources(
    args,
) -> tuple[TeaCocoDataset, Path, list[str], Path | None, list[ValEvalSource], list[float] | None]:
    """
    返回 (train_ds, train_ann_path, roots_used, merged_cache_dir_or_None, val_eval_sources)。
    train：单集用该集 train；多集用合并后的 merged_train.json。
    val 评测：每个根目录各一份原生 val（不合并）。
    """
    train_mosaic_p = args.aug_det_mosaic_p if args.aug_preset == "detection" else 0.0
    roots: list[Path] = list(args.datasets)

    val_sources: list[ValEvalSource] = []
    for root in roots:
        _, val_ann = _default_coco_train_val_paths(root)
        val_sources.append(
            ValEvalSource(
                name=root.name,
                root=root,
                dataset=TeaCocoDataset(root, val_ann, mosaic_p=0.0),
                ann_path=val_ann,
            )
        )

    if len(roots) == 1:
        if args.dataset_ratios:
            print("提示: 仅 1 个 --datasets 时忽略 --dataset_ratios。")
        train_ann, _ = _default_coco_train_val_paths(roots[0])
        train_ds = TeaCocoDataset(roots[0], train_ann, mosaic_p=train_mosaic_p)
        return train_ds, train_ann, [str(roots[0])], None, val_sources, None

    fingerprint = hashlib.md5("\n".join(str(r.resolve()) for r in roots).encode("utf-8")).hexdigest()[:16]
    merged_dir = args.output_dir / "merged_coco_cache" / fingerprint
    train_ann_m, _val_ann_m, train_weights = merge_coco_roots_for_training(
        roots, merged_dir, dataset_ratios=args.dataset_ratios
    )
    dummy_root = _SCRIPT_DIR
    train_ds = TeaCocoDataset(dummy_root, train_ann_m, mosaic_p=train_mosaic_p, sample_weights=train_weights)
    return train_ds, train_ann_m, [str(r) for r in roots], merged_dir, val_sources, train_weights


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
) -> v2.Compose:
    """对齐常见 RT-DETR/DEIM 式 train dataloader：几何 + 光度，Resize 640 后再交给 HF 做 ImageNet 归一化。"""
    return v2.Compose(
        [
            v2.RandomPhotometricDistort(p=photometric_p),
            v2.RandomZoomOut(fill=zoomout_fill, p=zoomout_p),
            v2.RandomApply([v2.RandomIoUCrop()], p=iou_crop_p),
            v2.SanitizeBoundingBoxes(min_size=1.0),
            v2.RandomHorizontalFlip(p=flip_p),
            v2.Resize(size=(640, 640)),
            v2.SanitizeBoundingBoxes(min_size=1.0),
        ]
    )


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
    """原图分辨率：水平翻转 + ColorJitter（与 aug_preset=simple 对应）。"""
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
    aug_preset: str,
    simple_flip_p: float,
    simple_color_p: float,
    det_photometric_p: float,
    det_zoomout_fill: int | float,
    det_zoomout_p: float,
    det_iou_crop_p: float,
    det_flip_p: float,
):
    color_jitter = T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02)
    detection_tf: v2.Compose | None = None
    if aug_preset == "detection":
        detection_tf = build_detection_augment(
            photometric_p=det_photometric_p,
            zoomout_fill=det_zoomout_fill,
            zoomout_p=det_zoomout_p,
            iou_crop_p=det_iou_crop_p,
            flip_p=det_flip_p,
        )

    def collate(batch):
        images = []
        annotations = []
        for img, image_id, target in batch:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tgt = list(target)
            if aug_preset == "simple":
                img, tgt = _apply_train_augment_simple(
                    img, tgt, flip_p=simple_flip_p, color_p=simple_color_p, color_jitter=color_jitter
                )
            elif aug_preset == "detection":
                assert detection_tf is not None
                w, h = img.size
                im_t = v2.functional.to_image(img)
                tdict = _coco_anns_to_tv_target(tgt, image_height=h, image_width=w)
                im_t2, t2 = detection_tf(im_t, tdict)
                img = TVF.to_pil_image(im_t2)
                tgt = _tv_target_to_coco_anns(t2)
            elif aug_preset == "none":
                pass
            else:
                raise ValueError(f"未知 aug_preset: {aug_preset}")
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
    ):
        self._coco = CocoDetection(str(root), str(ann_file))
        self.mosaic_p = float(mosaic_p)
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self._coco)

    def __getitem__(self, idx: int):
        if self.mosaic_p > 0.0 and len(self._coco) >= 4 and torch.rand(()).item() < self.mosaic_p:
            return self._getitem_mosaic(idx)
        img, target = self._coco[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")
        image_id = self._coco.ids[idx]
        return img, image_id, list(target)

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
            return img, self._coco.ids[idx], list(target)

        idxs = [idx, others[0], others[1], others[2]]
        quads: list[tuple[Image.Image, list]] = []
        for j in idxs:
            img, target = self._coco[j]
            if img.mode != "RGB":
                img = img.convert("RGB")
            quads.append((img, list(target)))
        mos_img, mos_tgt = _mosaic_pil_coco_quadrants(quads)
        return mos_img, self._coco.ids[idx], mos_tgt


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
    """推理用：只做 resize/normalize，不读标注。"""
    images = []
    target_sizes = []
    image_ids = []
    for img, image_id, _ in batch:
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)
        w, h = img.size
        target_sizes.append((h, w))
        image_ids.append(image_id)
    return images, torch.tensor(target_sizes, dtype=torch.int64), image_ids


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
    for images, target_sizes_cpu, image_ids in loader:
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
        for img_id, res in zip(image_ids, results):
            scores = res["scores"]
            labels = res["labels"]
            boxes = res["boxes"]
            for s, lab, box in zip(scores.tolist(), labels.tolist(), boxes):
                predictions.append(
                    {
                        "image_id": int(img_id),
                        "category_id": int(lab),
                        "bbox": _xyxy_to_xywh(box.cpu()),
                        "score": float(s),
                    }
                )

    if not predictions:
        first_id = int(coco_gt.getImgIds()[0])
        predictions.append(
            {
                "image_id": first_id,
                "category_id": 0,
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


def main():
    args = parse_args()
    args.datasets = [_resolve_cli_path(p) for p in args.datasets]
    if not args.datasets:
        raise ValueError("--datasets 至少指定一个 COCO 数据集根目录")
    args.output_dir = _resolve_cli_path(args.output_dir)
    if args.resume_from is not None:
        args.resume_from = _resolve_cli_path(args.resume_from)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.warmup_epochs < 0:
        raise ValueError("--warmup_epochs 不能为负数")

    start_epoch = 0
    resume_training_state: dict | None = None
    if args.resume_from is not None:
        resume_dir = args.resume_from
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"--resume_from 不是目录: {resume_dir}")
        processor = AutoImageProcessor.from_pretrained(str(resume_dir))
        model = Deimv2ForObjectDetection.from_pretrained(str(resume_dir))
        model.config.id2label = {0: "I", 1: "Y"}
        model.config.label2id = {"I": 0, "Y": 1}
        ts_path = _load_training_state_path(resume_dir)
        if ts_path is not None:
            resume_training_state = torch.load(ts_path, map_location="cpu", weights_only=False)
            start_epoch = int(resume_training_state["epoch"])
        else:
            inferred = _infer_completed_epoch(resume_dir)
            start_epoch = int(inferred) if inferred is not None else 0
    else:
        pretrained_hub = resolve_pretrained_hub_id(args.pretrained, args.pretrained_id)
        print(f"从 HuggingFace Hub 加载预训练: {pretrained_hub}")
        processor = AutoImageProcessor.from_pretrained(pretrained_hub)
        model = Deimv2ForObjectDetection.from_pretrained(
            pretrained_hub,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        model.config.id2label = {0: "I", 1: "Y"}
        model.config.label2id = {"I": 0, "Y": 1}

    load_source = str(args.resume_from) if args.resume_from is not None else resolve_pretrained_hub_id(
        args.pretrained, args.pretrained_id
    )

    apply_train_mode(model, args.train_mode, args.loss_bbox_scale)
    apply_dinov3_backbone_last_n_unfreeze(model, args.unfreeze_backbone_last_n)
    print(f"train_mode={args.train_mode}，可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    train_mosaic_p = args.aug_det_mosaic_p if args.aug_preset == "detection" else 0.0
    print(
        f"训练数据增强 preset={args.aug_preset}"
        + (
            f"（distort_p={args.aug_det_photometric_p}, zoomout_fill={args.aug_det_zoomout_fill}, "
            f"zoomout_p={args.aug_det_zoomout_p}, iou_crop_p={args.aug_det_iou_crop_p}, flip_p={args.aug_det_flip_p}, "
            f"mosaic_p={train_mosaic_p:g}）"
            if args.aug_preset == "detection"
            else (
                f"（flip_p={args.aug_simple_flip_p}, color_p={args.aug_simple_color_p}）"
                if args.aug_preset == "simple"
                else ""
            )
        )
    )

    train_ds, train_ann, dataset_roots, merged_coco_cache, val_eval_sources, train_sample_weights = (
        _build_train_val_sources(args)
    )
    print(f"训练数据: {'合并 train' if len(args.datasets) > 1 else '单集'}，共 {len(args.datasets)} 个数据集根目录:")
    for i, r in enumerate(dataset_roots):
        print(f"  [{i}] {r}")
    if merged_coco_cache is not None:
        print(f"合并 COCO 缓存目录: {merged_coco_cache}")
    print("每轮 val mAP 将按以下子集分别评测" + ("并求平均" if len(val_eval_sources) > 1 else "") + ":")
    for src in val_eval_sources:
        print(f"  - {src.name} ({src.ann_path})")

    if train_mosaic_p > 0 and len(train_ds) < 4:
        print("警告: --aug_det_mosaic_p>0 但训练集长度 <4，Mosaic 分支不会触发。")

    device = torch.device(args.device)
    model.to(device)

    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        coco_gt_train = COCO(str(train_ann))
    map_bs = args.map_batch_size if args.map_batch_size is not None else args.batch_size

    train_collate = make_train_collate_fn(
        processor,
        aug_preset=args.aug_preset,
        simple_flip_p=args.aug_simple_flip_p,
        simple_color_p=args.aug_simple_color_p,
        det_photometric_p=args.aug_det_photometric_p,
        det_zoomout_fill=args.aug_det_zoomout_fill,
        det_zoomout_p=args.aug_det_zoomout_p,
        det_iou_crop_p=args.aug_det_iou_crop_p,
        det_flip_p=args.aug_det_flip_p,
    )

    if train_sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(train_sample_weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=train_collate,
            pin_memory=device.type == "cuda",
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=train_collate,
            pin_memory=device.type == "cuda",
        )

    if args.backbone_lr_decay <= 0:
        raise ValueError("--backbone_lr_decay 必须 > 0")
    lr_backbone_eff = args.lr_backbone if args.lr_backbone is not None else args.lr * 0.125
    param_groups, used_layerwise = build_adamw_param_groups(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        unfreeze_backbone_last_n=args.unfreeze_backbone_last_n,
        lr_backbone=lr_backbone_eff,
        backbone_lr_decay=args.backbone_lr_decay,
    )
    opt = AdamW(param_groups)
    if used_layerwise:
        nl = _dinov3_vit_num_hidden_layers(model)
        print(
            f"优化器: 因 unfreeze_backbone_last_n={args.unfreeze_backbone_last_n}>0，对 DINOv3 可训练 backbone 分层 lr"
            f"（ViT block 数={nl}），lr_backbone={lr_backbone_eff:g}，由深到浅每层×{args.backbone_lr_decay:g}；"
            f"检测头/neck 等 lr={args.lr:g}；weight_decay={args.weight_decay:g}（backbone 内 norm/bias 等 wd=0），"
            f"param_groups={len(param_groups)}"
        )
        backbone_trainable = any(
            (".conv_encoder.backbone." in n or ".conv_encoder.model." in n)
            for n, p in model.named_parameters()
            if p.requires_grad
        )
        if not backbone_trainable:
            print(
                "警告: 已请求骨干分层 lr，但当前无可训练 conv_encoder 骨干参数（请确认 unfreeze 是否生效）。"
            )
    else:
        print(f"优化器: AdamW lr={args.lr:g}, weight_decay={args.weight_decay:g}, param_groups={len(param_groups)}")

    opt_resume = "none"  # full | named | fresh | none
    if resume_training_state is not None:
        if "optimizer" in resume_training_state:
            try:
                opt.load_state_dict(resume_training_state["optimizer"])
                opt_resume = "full"
            except Exception as exc:
                print(f"警告: 未能整表加载优化器 state_dict（param_groups 可能已变）: {exc}")
                n_named = apply_adamw_named_state(
                    opt, model, resume_training_state.get("optimizer_named_state")
                )
                opt_resume = "named" if n_named > 0 else "fresh"
                if n_named > 0:
                    print(
                        f"提示: 已按参数名合并 AdamW 动量到 {n_named} 个参数；"
                        "新解冻或新分组中的参数无历史动量。"
                    )
                else:
                    print("警告: checkpoint 中无可用 optimizer_named_state（旧格式），优化器动量已重新初始化。")
                    print(
                        "提示: 自本脚本起保存的 training_state.pt 会附带 optimizer_named_state；"
                        "若需今后再改 param_groups/解冻层数并续接动量，请从含该字段的 checkpoint 续训。"
                    )
        if resume_training_state.get("rng_cpu") is not None:
            torch.random.set_rng_state(resume_training_state["rng_cpu"])
        if resume_training_state.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume_training_state["rng_cuda"])

    if args.resume_from is not None:
        if resume_training_state is not None:
            if opt_resume == "full":
                what = "模型、优化器与 RNG"
            elif opt_resume == "named":
                what = "模型、RNG，以及按参数名合并的 AdamW 动量（param_groups 与保存时不一致）"
            else:
                what = "模型与 RNG（优化器为当前超参下的新 AdamW）"
            print(
                f"已从 {args.resume_from} 恢复{what}，"
                f"已完成 epoch {start_epoch}，将从 epoch {start_epoch + 1} 继续（总目标 epoch 数由 --epochs 指定）"
            )
        else:
            print(
                f"已从 {args.resume_from} 恢复模型权重（无 training_state.pt），"
                f"优化器重新初始化；推断已完成 epoch {start_epoch}，将从 epoch {start_epoch + 1} 继续"
            )

    next_epoch = start_epoch + 1
    if next_epoch > args.epochs:
        print(f"已完成 epoch {start_epoch}，且 >= --epochs {args.epochs}，无需继续训练。")
        return

    sync_optimizer_param_group_metadata(
        opt,
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        unfreeze_backbone_last_n=args.unfreeze_backbone_last_n,
        lr_backbone=lr_backbone_eff,
        backbone_lr_decay=args.backbone_lr_decay,
    )
    if args.warmup_epochs > 0:
        if any(g.get("is_backbone") for g in opt.param_groups):
            print(
                f"骨干线性热身: {args.warmup_epochs} 个 epoch（epoch 1 时 mult=0，epoch {args.warmup_epochs} 时 mult=1），"
                "仅 conv_encoder 骨干参数组；检测头等保持 base_lr。"
            )
        else:
            print("提示: --warmup_epochs>0 但当前无可训练骨干参数，热身未生效。")

    for epoch in range(next_epoch, args.epochs + 1):
        m_warm = apply_backbone_linear_warmup_lrs(opt, epoch, args.warmup_epochs)
        model.train()
        running = 0.0
        running_cls = 0.0
        running_l1_bbox = 0.0
        running_giou = 0.0
        n_batches = 0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = move_labels_to_device(batch["labels"], device)
            kwargs = {"pixel_values": pixel_values, "labels": labels}
            if "pixel_mask" in batch:
                kwargs["pixel_mask"] = batch["pixel_mask"].to(device)

            opt.zero_grad()
            out = model(**kwargs)
            loss = out.loss
            loss.backward()
            opt.step()

            running += float(loss.detach())
            ld = getattr(out, "loss_dict", None)
            running_cls += _sum_weighted_cls_loss(ld)
            running_l1_bbox += _sum_weighted_l1_bbox_loss(ld)
            running_giou += _sum_weighted_giou_loss(ld)
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        train_loss_cls = running_cls / max(n_batches, 1)
        train_loss_l1_bbox = running_l1_bbox / max(n_batches, 1)
        train_loss_giou = running_giou / max(n_batches, 1)

        map_train = evaluate_coco_bbox_map(
            model,
            processor,
            train_ds,
            coco_gt_train,
            device,
            batch_size=map_bs,
            score_threshold=args.map_score_threshold,
            num_workers=args.num_workers,
        )
        map_val_per, map_val_mean = evaluate_val_maps_per_dataset(
            model,
            processor,
            val_eval_sources,
            device,
            batch_size=map_bs,
            score_threshold=args.map_score_threshold,
            num_workers=args.num_workers,
        )
        map_val = map_val_mean if len(val_eval_sources) > 1 else next(iter(map_val_per.values()))

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} | cls≈{train_loss_cls:.4f} | "
            f"bbox(L1)≈{train_loss_l1_bbox:.4f} | giou≈{train_loss_giou:.4f}"
            + (f" | bb_warmup={m_warm:.3f}" if args.warmup_epochs > 0 else "")
        )
        print(
            f"  train mAP={map_train['bbox_mAP']:.4f} mAP50={map_train['bbox_mAP_50']:.4f} "
            f"mAP75={map_train['bbox_mAP_75']:.4f} AR100={map_train['bbox_mAR_100']:.4f}"
        )

        save_dir = args.output_dir / f"checkpoint-epoch{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_cls_weighted": train_loss_cls,
            "train_loss_bbox_l1_weighted": train_loss_l1_bbox,
            "train_loss_giou_weighted": train_loss_giou,
            "map_score_threshold": args.map_score_threshold,
            "train_map": map_train,
            "val_map": map_val,
            "val_map_per_dataset": map_val_per,
            "val_map_mean": map_val_mean if len(val_eval_sources) > 1 else None,
            "train_mode": args.train_mode,
            "unfreeze_backbone_last_n": args.unfreeze_backbone_last_n,
            "warmup_epochs": args.warmup_epochs,
            "backbone_warmup_mult": m_warm,
            "pretrained": args.pretrained,
            "pretrained_id": args.pretrained_id,
            "load_source": load_source,
            "dataset_roots": dataset_roots,
            "dataset_ratios": list(args.dataset_ratios) if args.dataset_ratios else None,
            "train_sample_weights_enabled": train_sample_weights is not None,
            "merged_coco_cache": str(merged_coco_cache) if merged_coco_cache is not None else None,
            "optimizer": {
                "layerwise_dinov3_backbone_lr": used_layerwise,
                "lr": args.lr,
                "lr_backbone": lr_backbone_eff if used_layerwise else None,
                "backbone_lr_decay": args.backbone_lr_decay if used_layerwise else None,
                "weight_decay": args.weight_decay,
                "param_groups": len(param_groups),
            },
            "augmentation": {
                "preset": args.aug_preset,
                "simple": {"flip_p": args.aug_simple_flip_p, "color_p": args.aug_simple_color_p},
                "detection": {
                    "photometric_p": args.aug_det_photometric_p,
                    "zoomout_fill": args.aug_det_zoomout_fill,
                    "zoomout_p": args.aug_det_zoomout_p,
                    "iou_crop_p": args.aug_det_iou_crop_p,
                    "flip_p": args.aug_det_flip_p,
                    "mosaic_p": train_mosaic_p,
                },
            },
            "loss_weights": {
                "mal": model.config.weight_loss_mal,
                "bbox": model.config.weight_loss_bbox,
                "giou": model.config.weight_loss_giou,
                "fgl": model.config.weight_loss_fgl,
                "ddf": model.config.weight_loss_ddf,
            },
        }
        with open(save_dir / "train_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.save(
            {
                "epoch": epoch,
                "optimizer": opt.state_dict(),
                "optimizer_named_state": pack_adamw_named_state(opt, model),
                "rng_cpu": torch.random.get_rng_state(),
                "rng_cuda": rng_cuda,
                "unfreeze_backbone_last_n": args.unfreeze_backbone_last_n,
                "warmup_epochs": args.warmup_epochs,
                "pretrained": args.pretrained,
                "pretrained_id": args.pretrained_id,
                "load_source": load_source,
            },
            save_dir / "training_state.pt",
        )

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.save(
        {
            "epoch": args.epochs,
            "optimizer": opt.state_dict(),
            "optimizer_named_state": pack_adamw_named_state(opt, model),
            "rng_cpu": torch.random.get_rng_state(),
            "rng_cuda": rng_cuda,
            "unfreeze_backbone_last_n": args.unfreeze_backbone_last_n,
            "warmup_epochs": args.warmup_epochs,
            "pretrained": args.pretrained,
            "pretrained_id": args.pretrained_id,
            "load_source": load_source,
        },
        final_dir / "training_state.pt",
    )
    print(f"训练结束，模型与 training_state.pt 已保存到 {final_dir}")


if __name__ == "__main__":
    main()
