"""
使用 DEIMv2-S（DINOv3-S）在茶叶 COCO 数据集上做迁移学习。

使用 HuggingFace Transformers中的DEIMv2_DINOv3_S_COCO权重，便于训练与保存。

默认（--train_mode backbone_frozen）：只冻结 conv_encoder 内的 backbone（DINOv3 为
model.conv_encoder.backbone.*，CNN 骨干为 model.conv_encoder.model.*）；neck（如 STA、
fusion_proj）、HybridEncoder、Decoder、分类与框回归等其余参数全部训练，并使用完整
检测损失。可用 --loss_bbox_scale 放大 bbox/giou 权重。

其它模式：--train_mode heads_only 仅训练各检测头（仍冻 encoder/decoder 主体）；
--train_mode classification_only 为仅分类头且 bbox 类损失置 0（易 mAP 接近 0）。

断点续训：使用 --resume_from 指向某次保存目录（如 checkpoint-epoch5）。若该目录含
training_state.pt（每轮保存时会写入），将恢复优化器与 RNG；否则仅加载模型权重，
并根据 train_metrics.json 或目录名推断下一 epoch。

依赖：torch, torchvision, transformers, scipy（Hungarian 匹配需要）, pycocotools（验证 mAP）

训练数据增强在 collate 中完成，由 --aug_preset 选择：
  detection（默认）：RandomPhotometricDistort、RandomZoomOut、RandomIoUCrop（p 可配）、
  SanitizeBoundingBoxes、RandomHorizontalFlip、Resize(640)，再交给 HF processor 归一化；
  simple：仅水平翻转 + ColorJitter；none：不增强。验证/mAP 始终无增强。
"""

from __future__ import annotations

import argparse
import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TVF
from torchvision.transforms import v2
from transformers import AutoImageProcessor, Deimv2ForObjectDetection
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

PRETRAINED_ID = "harshaljanjani/DEIMv2_DINOv3_S_COCO_Transformers"


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
    def __init__(self, root: Path, ann_file: Path):
        self._coco = CocoDetection(str(root), str(ann_file))

    def __len__(self):
        return len(self._coco)

    def __getitem__(self, idx: int):
        img, target = self._coco[idx]
        if img.mode != "RGB":
            img = img.convert("RGB")
        image_id = self._coco.ids[idx]
        return img, image_id, target


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root",
        type=Path,
        default=Path(r"E:\teaDetector\datasets\TeaLeavesDatasets_split_lr"),
    )
    p.add_argument("--output_dir", type=Path, default=Path(r"E:\teaDetector\outputs\deimv2_s_tealeaves"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
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
        help="从某次保存的目录恢复（含 config.json、模型权重）。若同目录有 training_state.pt 则一并恢复优化器并从下一 epoch 继续",
    )
    p.add_argument(
        "--aug_preset",
        type=str,
        choices=("detection", "simple", "none"),
        default="detection",
        help="detection=torchvision v2 光度+ZoomOut+IoUCrop+翻转+Resize640（参考 RT-DETR 式配置）；"
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
        default=0.5,
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
        default=0.5,
        help="aug_preset=detection：RandomZoomOut 概率（torchvision 默认 0.5）",
    )
    p.add_argument(
        "--aug_det_iou_crop_p",
        type=float,
        default=0.8,
        help="aug_preset=detection：RandomIoUCrop 以 RandomApply 包裹的应用概率",
    )
    p.add_argument(
        "--aug_det_flip_p",
        type=float,
        default=0.5,
        help="aug_preset=detection：RandomHorizontalFlip 概率",
    )
    return p.parse_args()


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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    resume_training_state: dict | None = None
    if args.resume_from is not None:
        resume_dir = args.resume_from.expanduser().resolve()
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
        processor = AutoImageProcessor.from_pretrained(PRETRAINED_ID)
        model = Deimv2ForObjectDetection.from_pretrained(
            PRETRAINED_ID,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        model.config.id2label = {0: "I", 1: "Y"}
        model.config.label2id = {"I": 0, "Y": 1}

    apply_train_mode(model, args.train_mode, args.loss_bbox_scale)
    print(f"train_mode={args.train_mode}，可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(
        f"训练数据增强 preset={args.aug_preset}"
        + (
            f"（distort_p={args.aug_det_photometric_p}, zoomout_fill={args.aug_det_zoomout_fill}, "
            f"zoomout_p={args.aug_det_zoomout_p}, iou_crop_p={args.aug_det_iou_crop_p}, flip_p={args.aug_det_flip_p}）"
            if args.aug_preset == "detection"
            else (
                f"（flip_p={args.aug_simple_flip_p}, color_p={args.aug_simple_color_p}）"
                if args.aug_preset == "simple"
                else ""
            )
        )
    )

    train_ds = TeaCocoDataset(args.data_root, args.data_root / "annotations" / "train.json")
    train_ann = args.data_root / "annotations" / "train.json"
    val_ann = args.data_root / "annotations" / "val.json"
    val_ds = TeaCocoDataset(args.data_root, val_ann)

    device = torch.device(args.device)
    model.to(device)

    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        coco_gt_train = COCO(str(train_ann))
        coco_gt_val = COCO(str(val_ann))
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

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collate,
        pin_memory=device.type == "cuda",
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=args.lr)

    if resume_training_state is not None:
        if "optimizer" in resume_training_state:
            opt.load_state_dict(resume_training_state["optimizer"])
        if resume_training_state.get("rng_cpu") is not None:
            torch.random.set_rng_state(resume_training_state["rng_cpu"])
        if resume_training_state.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume_training_state["rng_cuda"])

    if args.resume_from is not None:
        if resume_training_state is not None:
            print(
                f"已从 {args.resume_from} 恢复模型、优化器与 RNG，"
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

    for epoch in range(next_epoch, args.epochs + 1):
        model.train()
        running = 0.0
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
            n_batches += 1

        train_loss = running / max(n_batches, 1)

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
        map_val = evaluate_coco_bbox_map(
            model,
            processor,
            val_ds,
            coco_gt_val,
            device,
            batch_size=map_bs,
            score_threshold=args.map_score_threshold,
            num_workers=args.num_workers,
        )

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} | "
            f"train mAP={map_train['bbox_mAP']:.4f} mAP50={map_train['bbox_mAP_50']:.4f} "
            f"mAP75={map_train['bbox_mAP_75']:.4f} AR100={map_train['bbox_mAR_100']:.4f} | "
            f"val mAP={map_val['bbox_mAP']:.4f} mAP50={map_val['bbox_mAP_50']:.4f} "
            f"mAP75={map_val['bbox_mAP_75']:.4f} AR100={map_val['bbox_mAR_100']:.4f}"
        )

        save_dir = args.output_dir / f"checkpoint-epoch{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "map_score_threshold": args.map_score_threshold,
            "train_map": map_train,
            "val_map": map_val,
            "train_mode": args.train_mode,
            "augmentation": {
                "preset": args.aug_preset,
                "simple": {"flip_p": args.aug_simple_flip_p, "color_p": args.aug_simple_color_p},
                "detection": {
                    "photometric_p": args.aug_det_photometric_p,
                    "zoomout_fill": args.aug_det_zoomout_fill,
                    "zoomout_p": args.aug_det_zoomout_p,
                    "iou_crop_p": args.aug_det_iou_crop_p,
                    "flip_p": args.aug_det_flip_p,
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
                "rng_cpu": torch.random.get_rng_state(),
                "rng_cuda": rng_cuda,
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
            "rng_cpu": torch.random.get_rng_state(),
            "rng_cuda": rng_cuda,
        },
        final_dir / "training_state.pt",
    )
    print(f"训练结束，模型与 training_state.pt 已保存到 {final_dir}")


if __name__ == "__main__":
    main()
