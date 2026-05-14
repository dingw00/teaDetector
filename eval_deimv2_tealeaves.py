"""
对 train_deimv2_s_tealeaves.py 保存的 DEIMv2 HuggingFace 权重做评测。

指标与 eval_onnx_detector.py 一致：AP50、AP75、mAP@[0.50:0.95]（VOC 风格 PR / 逐 IoU AP 再平均），
IoU 阈值列表来自 onnx_eval_config.MAP_IOU_THRESHOLDS。该口径与训练脚本里 pycocotools 的 COCO mAP
不同，数值不宜与 bbox_mAP / bbox_mAP_50 等直接对比。

用法示例：
    python eval_deimv2_tealeaves.py --checkpoint E:\\teaDetector\\outputs\\deimv2_s_tealeaves\\final
    python eval_deimv2_tealeaves.py --checkpoint ... --conf 0.1 --nms 0.3 --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, Deimv2ForObjectDetection

import onnx_eval_config as cfg


@dataclass
class CocoDataset:
    name: str
    ann_path: Path
    image_dir: Path
    images: list[dict]
    gt_by_image: dict[int, list[dict]]
    category_names: dict[int, str]


def load_coco(name: str, ann_path: Path, image_dir: Path) -> CocoDataset:
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    category_names = {cat["id"]: cat.get("name", str(cat["id"])) for cat in coco.get("categories", [])}
    gt_by_image = {img["id"]: [] for img in coco["images"]}

    for ann in coco.get("annotations", []):
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0 or ann.get("iscrowd", 0):
            continue
        gt_by_image.setdefault(ann["image_id"], []).append(
            {
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "category_id": ann["category_id"],
                "matched": False,
            }
        )

    return CocoDataset(
        name=name,
        ann_path=ann_path,
        image_dir=image_dir,
        images=coco["images"],
        gt_by_image=gt_by_image,
        category_names=category_names,
    )


def resolve_image_path(image_dir: Path, file_name: str) -> Path:
    path = image_dir / file_name
    if path.exists():
        return path

    matches = list(image_dir.rglob(Path(file_name).name))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"找不到图片：{file_name}，搜索目录：{image_dir}")


def make_safe_name(text: str) -> str:
    chars = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-6)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_xyxy(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= threshold]
    return keep


def voc_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_ap_for_iou(preds_by_image, gt_by_image, categories, iou_thr: float):
    aps = []
    for cat_id in categories:
        preds = []
        total_gts = 0
        gt_used = {}

        for image_id, gts in gt_by_image.items():
            cat_gts = [g for g in gts if g["category_id"] == cat_id]
            total_gts += len(cat_gts)
            gt_used[image_id] = np.zeros(len(cat_gts), dtype=bool)

            for pred in preds_by_image.get(image_id, []):
                if pred["category_id"] == cat_id:
                    preds.append((image_id, pred["score"], np.array(pred["bbox"], dtype=np.float32)))

        if total_gts == 0:
            continue

        preds.sort(key=lambda x: x[1], reverse=True)
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for i, (image_id, _, pred_box) in enumerate(preds):
            cat_gts = [g for g in gt_by_image.get(image_id, []) if g["category_id"] == cat_id]
            gt_boxes = np.array([g["bbox"] for g in cat_gts], dtype=np.float32)

            if gt_boxes.size == 0:
                fp[i] = 1.0
                continue

            ious = iou_xyxy(pred_box, gt_boxes)
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])

            if best_iou >= iou_thr and not gt_used[image_id][best_idx]:
                tp[i] = 1.0
                gt_used[image_id][best_idx] = True
            else:
                fp[i] = 1.0

        if len(preds) == 0:
            aps.append(0.0)
            continue

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(total_gts, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
        aps.append(voc_ap(recalls, precisions))

    return float(np.mean(aps)) if aps else 0.0


def compute_map(preds_by_image, dataset: CocoDataset):
    categories = sorted(dataset.category_names.keys())
    ap_by_iou = {
        thr: compute_ap_for_iou(preds_by_image, dataset.gt_by_image, categories, thr)
        for thr in cfg.MAP_IOU_THRESHOLDS
    }
    return {
        "AP50": ap_by_iou.get(0.5, 0.0),
        "AP75": ap_by_iou.get(0.75, 0.0),
        "mAP50_95": float(np.mean(list(ap_by_iou.values()))) if ap_by_iou else 0.0,
        "ap_by_iou": ap_by_iou,
    }


def draw_boxes(image_bgr, preds, gts, category_names, save_path: Path):
    image = image_bgr.copy()

    for gt in gts:
        x1, y1, x2, y2 = [int(round(v)) for v in gt["bbox"]]
        name = category_names.get(gt["category_id"], str(gt["category_id"]))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            image,
            f"GT:{name}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )

    for pred in preds:
        x1, y1, x2, y2 = [int(round(v)) for v in pred["bbox"]]
        name = category_names.get(pred["category_id"], str(pred["category_id"]))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.putText(
            image,
            f"P:{name} {pred['score']:.2f}",
            (x1, min(image.shape[0] - 5, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 80, 255),
            1,
            cv2.LINE_AA,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image)


def postprocess_nms_per_class(
    scores: torch.Tensor, labels: torch.Tensor, boxes: torch.Tensor, nms_thres: float
) -> list[dict]:
    """boxes: xyxy on original image."""
    preds = []
    for s, lab, box in zip(scores.tolist(), labels.tolist(), boxes.cpu().float().numpy()):
        x1, y1, x2, y2 = (float(v) for v in box)
        if x2 <= x1 or y2 <= y1:
            continue
        preds.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(s),
                "category_id": int(lab),
            }
        )

    if not preds or nms_thres >= 1.0:
        return sorted(preds, key=lambda p: p["score"], reverse=True)

    final_preds = []
    categories = sorted({p["category_id"] for p in preds})
    for cat_id in categories:
        cat_preds = [p for p in preds if p["category_id"] == cat_id]
        cat_boxes = np.array([p["bbox"] for p in cat_preds], dtype=np.float32)
        cat_scores = np.array([p["score"] for p in cat_preds], dtype=np.float32)
        keep = nms_numpy(cat_boxes, cat_scores, nms_thres)
        final_preds.extend(cat_preds[i] for i in keep)

    return sorted(final_preds, key=lambda p: p["score"], reverse=True)


class CocoImageListDataset(Dataset):
    def __init__(self, coco_ds: CocoDataset):
        self.coco_ds = coco_ds

    def __len__(self):
        return len(self.coco_ds.images)

    def __getitem__(self, idx: int):
        img_info = self.coco_ds.images[idx]
        path = resolve_image_path(self.coco_ds.image_dir, img_info["file_name"])
        image = Image.open(path).convert("RGB")
        w, h = image.size
        return image, torch.tensor([h, w], dtype=torch.int64), int(img_info["id"])


def collate_batch(batch):
    images, target_sizes, image_ids = zip(*batch)
    return list(images), torch.stack(target_sizes, dim=0), list(image_ids)


def parse_args():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="DEIMv2 (HF) 茶叶模型评测，指标对齐 eval_onnx_detector")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "outputs" / "deimv2_s_tealeaves" / "final",
        help="save_pretrained 目录（含 config.json、模型权重与 preprocessor）",
    )
    p.add_argument(
        "--data_root",
        type=Path,
        default=root / "datasets" / "TeaLeavesDatasets_split_lr",
    )
    p.add_argument("--train_ann", type=Path, default=None, help="默认 data_root/annotations/train.json")
    p.add_argument("--val_ann", type=Path, default=None, help="默认 data_root/annotations/val.json")
    p.add_argument("--output_dir", type=Path, default=root / "outputs" / "deimv2_hf_eval")
    p.add_argument("--conf", type=float, default=0.05, help="score 阈值（与训练验证 map_score_threshold 默认一致）")
    p.add_argument("--nms", type=float, default=cfg.NMS_THRESHOLD, help="按类 NMS IoU 阈值，与 ONNX 脚本后处理一致")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vis_num", type=int, default=cfg.VIS_NUM_IMAGES)
    p.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--eval_train", action="store_true", help="同时评测 train.json（默认只评 val）")
    return p.parse_args()


def build_run_name(args, model_stem: str) -> str:
    if args.run_name:
        return make_safe_name(args.run_name)

    dataset_name = make_safe_name(args.data_root.name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{model_stem}_{dataset_name}_conf{args.conf:g}_nms{args.nms:g}"


@torch.no_grad()
def evaluate_dataset(
    model: Deimv2ForObjectDetection,
    processor,
    dataset: CocoDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    conf: float,
    nms: float,
    output_dir: Path,
    vis_num: int,
    seed: int,
):
    print(f"\n评测 {dataset.name}: {dataset.ann_path}")
    ds = CocoImageListDataset(dataset)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    preds_by_image: dict[int, list] = {}
    for images, target_sizes_cpu, image_ids in tqdm(loader, desc=dataset.name):
        enc = processor(images=images, return_tensors="pt")
        pixel_values = enc["pixel_values"].to(device)
        target_sizes = target_sizes_cpu.to(device)
        kwargs = {"pixel_values": pixel_values}
        if "pixel_mask" in enc:
            kwargs["pixel_mask"] = enc["pixel_mask"].to(device)

        outputs = model(**kwargs)
        results = processor.post_process_object_detection(outputs, threshold=conf, target_sizes=target_sizes)

        for image_id, res in zip(image_ids, results):
            preds_by_image[image_id] = postprocess_nms_per_class(
                res["scores"], res["labels"], res["boxes"], nms_thres=nms
            )

    metrics = compute_map(preds_by_image, dataset)
    print(f"{dataset.name} 指标:")
    print(f"  AP50      = {metrics['AP50']:.4f}")
    print(f"  AP75      = {metrics['AP75']:.4f}")
    print(f"  mAP50-95  = {metrics['mAP50_95']:.4f}")

    rng = random.Random(seed)
    samples = rng.sample(dataset.images, k=min(vis_num, len(dataset.images)))
    vis_dir = output_dir / dataset.name
    for img_info in samples:
        path = resolve_image_path(dataset.image_dir, img_info["file_name"])
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise RuntimeError(f"无法读取图片：{path}")
        preds = preds_by_image.get(img_info["id"], [])
        gts = dataset.gt_by_image.get(img_info["id"], [])
        save_name = f"{Path(img_info['file_name']).stem}_pred_gt.jpg"
        draw_boxes(image_bgr, preds, gts, dataset.category_names, vis_dir / save_name)

    return metrics


def main():
    args = parse_args()
    train_ann = args.train_ann or (args.data_root / "annotations" / "train.json")
    val_ann = args.val_ann or (args.data_root / "annotations" / "val.json")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"找不到 checkpoint 目录：{args.checkpoint}")

    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(str(args.checkpoint))
    model = Deimv2ForObjectDetection.from_pretrained(str(args.checkpoint))
    model.to(device)
    model.eval()

    image_dir = args.data_root
    val_set = load_coco("val", val_ann, image_dir)
    train_set = load_coco("train", train_ann, image_dir) if args.eval_train else None

    model_stem = make_safe_name(args.checkpoint.name)
    run_name = build_run_name(args, model_stem)
    run_output_dir = args.output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=False)

    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}, batch_size={args.batch_size}, conf={args.conf}, nms={args.nms}")
    print(f"run_name: {run_name}")
    print(f"输出目录: {run_output_dir}")

    val_metrics = evaluate_dataset(
        model,
        processor,
        val_set,
        device,
        args.batch_size,
        args.num_workers,
        args.conf,
        args.nms,
        run_output_dir,
        args.vis_num,
        args.seed,
    )
    train_metrics = None
    if train_set is not None:
        train_metrics = evaluate_dataset(
            model,
            processor,
            train_set,
            device,
            args.batch_size,
            args.num_workers,
            args.conf,
            args.nms,
            run_output_dir,
            args.vis_num,
            args.seed + 1,
        )

    def _metrics_for_json(m: dict) -> dict:
        out = {k: v for k, v in m.items() if k != "ap_by_iou"}
        out["ap_by_iou"] = {str(k): float(v) for k, v in m["ap_by_iou"].items()}
        return out

    val_json = _metrics_for_json(val_metrics)
    train_json = _metrics_for_json(train_metrics) if train_metrics is not None else None
    metrics_block = {"val": val_json}
    if train_json is not None:
        metrics_block["train"] = train_json

    all_metrics = {
        "run": {
            "name": run_name,
            "output_dir": str(run_output_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "model": {
            "name": model_stem,
            "path": str(args.checkpoint),
        },
        "dataset": {
            "name": args.data_root.name,
            "image_dir": str(image_dir),
            "train_ann": str(train_ann),
            "val_ann": str(val_ann),
        },
        "eval_params": {
            "conf_threshold": args.conf,
            "nms_threshold": args.nms,
            "batch_size": args.batch_size,
            "device": str(device),
            "map_iou_thresholds": list(cfg.MAP_IOU_THRESHOLDS),
        },
        "metrics": metrics_block,
        "val": val_json,
    }
    if train_json is not None:
        all_metrics["train"] = train_json

    metrics_path = run_output_dir / f"metrics_{run_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n指标已保存: {metrics_path}")


if __name__ == "__main__":
    main()
