"""
茶叶目标检测统一评测：多模型 × 多数据集（各 train/val）。

指标：AP50、AP75、mAP@[0.50:0.95]（VOC 风格），IoU 列表见 eval_config.MAP_IOU_THRESHOLDS。

用法：
    python eval_tea.py
    python eval_tea.py --model onnx_models/a.onnx outputs/deimv2_s/final
    python eval_tea.py --conf 0.2 --nms 0.3 --val_only

配置：eval_config.py（DATASETS、DEFAULT_MODELS、阈值等）。
可视化抽样由 --seed 固定，不同模型对同一数据集/划分抽取相同图片。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

import eval_config as cfg

Backend = Literal["onnx", "checkpoint"]
MetricName = Literal["AP50", "AP75", "mAP50_95"]


@dataclass
class DatasetSpec:
    name: str
    image_dir: Path
    train_ann: Path
    val_ann: Path


@dataclass
class CocoDataset:
    name: str
    ann_path: Path
    image_dir: Path
    images: list[dict]
    gt_by_image: dict[int, list[dict]]
    gt_by_image_vis: dict[int, list[dict]]
    category_names: dict[int, str]


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="茶叶检测评测（多模型 × 多数据集）")
    p.add_argument(
        "--model",
        nargs="*",
        type=Path,
        default=None,
        help="一个或多个模型路径；默认 eval_config.DEFAULT_MODELS",
    )
    p.add_argument("--output_dir", type=Path, default=cfg.OUTPUT_DIR)
    p.add_argument("--input_size", type=int, default=cfg.INPUT_SIZE)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--nms", type=float, default=cfg.NMS_THRESHOLD)
    p.add_argument("--vis_num", type=int, default=cfg.VIS_NUM_IMAGES)
    p.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--val_only", action="store_true")
    p.add_argument("--no_plots", action="store_true", help="跳过对比图表生成")
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
    return p.parse_args(argv)


def resolve_dataset_specs() -> list[DatasetSpec]:
    if not cfg.DATASETS:
        raise ValueError("eval_config.DATASETS 为空，请至少配置一个数据集")
    specs = []
    for entry in cfg.DATASETS:
        root = Path(entry["root"])
        name = entry.get("name", root.name)
        specs.append(
            DatasetSpec(
                name=name,
                image_dir=root / "images",
                train_ann=root / "annotations" / "train.json",
                val_ann=root / "annotations" / "val.json",
            )
        )
    return specs


def detect_backend(model_path: Path) -> Backend:
    if model_path.suffix.lower() == ".onnx":
        return "onnx"
    return "checkpoint"


def default_conf(backend: Backend) -> float:
    return cfg.HF_CONF_THRESHOLD if backend == "checkpoint" else cfg.CONF_THRESHOLD


def resolve_torch_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_onnx_providers(device: str | None, cli_providers: list[str] | None) -> list[str]:
    """优先级：--providers > eval_config.ONNX_PROVIDERS > --device > 自动检测 CUDA。"""
    if cli_providers:
        return list(cli_providers)

    config_providers = getattr(cfg, "ONNX_PROVIDERS", None)
    if config_providers:
        return list(config_providers)

    import onnxruntime as ort

    available = ort.get_available_providers()
    dev = device.lower().strip() if device else None

    if dev == "cpu":
        return ["CPUExecutionProvider"]
    if dev == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise SystemExit(
                "已指定 DEVICE=cuda，但 onnxruntime 无 CUDAExecutionProvider。\n"
                "请安装 GPU 版：pip install onnxruntime-gpu，并确保 CUDA 驱动可用。"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def make_safe_name(text: str) -> str:
    chars = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def conf_tag_for_run_name(args, model_paths: list[Path]) -> str:
    if args.conf is not None:
        return f"{args.conf:g}"
    backends = {detect_backend(p) for p in model_paths}
    if len(backends) == 1:
        return f"{default_conf(next(iter(backends))):g}"
    return "auto"


def build_run_name(args, num_models: int, model_paths: list[Path]) -> str:
    if args.run_name:
        return make_safe_name(args.run_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    conf_tag = conf_tag_for_run_name(args, model_paths)
    return f"{ts}_m{num_models}_conf{conf_tag}_nms{args.nms:g}"


def vis_split_seed(base_seed: int, dataset_name: str, split: str) -> int:
    h = 0
    for ch in f"{dataset_name}:{split}":
        h = (h * 31 + ord(ch)) & 0x7FFFFFFF
    return (base_seed + h) % (2**31 - 1)


def load_coco(name: str, ann_path: Path, image_dir: Path) -> CocoDataset:
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    category_names = {cat["id"]: cat.get("name", str(cat["id"])) for cat in coco.get("categories", [])}
    gt_by_image = {img["id"]: [] for img in coco["images"]}
    gt_by_image_vis = {img["id"]: [] for img in coco["images"]}

    for ann in coco.get("annotations", []):
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            continue
        xyxy = [float(x), float(y), float(x + w), float(y + h)]
        cid = ann["category_id"]
        crowd = bool(ann.get("iscrowd", 0))
        iid = ann["image_id"]

        gt_by_image_vis.setdefault(iid, []).append({"bbox": xyxy, "category_id": cid, "iscrowd": crowd})
        if crowd:
            continue
        gt_by_image.setdefault(iid, []).append({"bbox": xyxy, "category_id": cid, "matched": False})

    return CocoDataset(
        name=name,
        ann_path=ann_path,
        image_dir=image_dir,
        images=coco["images"],
        gt_by_image=gt_by_image,
        gt_by_image_vis=gt_by_image_vis,
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


def pick_vis_image_infos(dataset: CocoDataset, vis_num: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return rng.sample(dataset.images, k=min(vis_num, len(dataset.images)))


def build_vis_plan(
    specs: list[DatasetSpec], vis_num: int, base_seed: int, val_only: bool
) -> dict[str, dict[str, list[dict]]]:
    """dataset -> split -> [{id, file_name}, ...]，全模型共用。"""
    plan: dict[str, dict[str, list[dict]]] = {}
    for spec in specs:
        plan[spec.name] = {}
        if not val_only:
            if spec.train_ann.exists():
                train_ds = load_coco("train", spec.train_ann, spec.image_dir)
                infos = pick_vis_image_infos(train_ds, vis_num, vis_split_seed(base_seed, spec.name, "train"))
                plan[spec.name]["train"] = [
                    {"id": int(i["id"]), "file_name": i["file_name"]} for i in infos
                ]
            else:
                print(f"警告: 未找到 {spec.name} train 标注，跳过 train 可视化抽样。")
        val_ds = load_coco("val", spec.val_ann, spec.image_dir)
        infos = pick_vis_image_infos(val_ds, vis_num, vis_split_seed(base_seed, spec.name, "val"))
        plan[spec.name]["val"] = [{"id": int(i["id"]), "file_name": i["file_name"]} for i in infos]
    return plan


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


def postprocess_nms_per_class(preds: list[dict], nms_thres: float) -> list[dict]:
    if not preds or nms_thres >= 1.0:
        return sorted(preds, key=lambda p: p["score"], reverse=True)
    final_preds = []
    for cat_id in sorted({p["category_id"] for p in preds}):
        cat_preds = [p for p in preds if p["category_id"] == cat_id]
        cat_boxes = np.array([p["bbox"] for p in cat_preds], dtype=np.float32)
        cat_scores = np.array([p["score"] for p in cat_preds], dtype=np.float32)
        keep = nms_numpy(cat_boxes, cat_scores, nms_thres)
        final_preds.extend(cat_preds[i] for i in keep)
    return sorted(final_preds, key=lambda p: p["score"], reverse=True)


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
            if float(ious[best_idx]) >= iou_thr and not gt_used[image_id][best_idx]:
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


def metrics_for_json(m: dict) -> dict:
    out = {k: v for k, v in m.items() if k != "ap_by_iou"}
    out["ap_by_iou"] = {str(k): float(v) for k, v in m["ap_by_iou"].items()}
    return out


def draw_boxes(image_bgr, preds, gts, category_names, save_path: Path):
    image = image_bgr.copy()
    for gt in gts:
        x1, y1, x2, y2 = [int(round(v)) for v in gt["bbox"]]
        name = category_names.get(gt["category_id"], str(gt["category_id"]))
        suffix = " [crowd]" if gt.get("iscrowd") else ""
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            image, f"GT:{name}{suffix}", (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA,
        )
    for pred in preds:
        x1, y1, x2, y2 = [int(round(v)) for v in pred["bbox"]]
        name = category_names.get(pred["category_id"], str(pred["category_id"]))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.putText(
            image, f"P:{name} {pred['score']:.2f}", (x1, min(image.shape[0] - 5, y2 + 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1, cv2.LINE_AA,
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image)


def visualize_by_ids(
    dataset: CocoDataset,
    preds_by_image: dict[int, list],
    vis_dir: Path,
    image_infos: list[dict],
):
    id_to_info = {int(i["id"]): i for i in image_infos}
    for image_id, img_info in id_to_info.items():
        path = resolve_image_path(dataset.image_dir, img_info["file_name"])
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise RuntimeError(f"无法读取图片：{path}")
        preds = preds_by_image.get(image_id, [])
        gts = dataset.gt_by_image_vis.get(image_id, [])
        save_name = f"{Path(img_info['file_name']).stem}_pred_gt.jpg"
        draw_boxes(image_bgr, preds, gts, dataset.category_names, vis_dir / save_name)


def print_metrics(dataset_label: str, metrics: dict):
    print(f"{dataset_label} 指标:")
    print(f"  AP50      = {metrics['AP50']:.4f}")
    print(f"  AP75      = {metrics['AP75']:.4f}")
    print(f"  mAP50-95  = {metrics['mAP50_95']:.4f}")


# ---------------------------------------------------------------------------
# ONNX
# ---------------------------------------------------------------------------


def preprocess_bgr(image_bgr: np.ndarray, input_size: int):
    orig_h, orig_w = image_bgr.shape[:2]
    ratio = min(float(input_size) / orig_w, float(input_size) / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2
    resized = cv2.resize(image_bgr, (new_w, new_h))
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = canvas.astype(np.float32) / 255.0
    tensor = np.transpose(blob, (2, 0, 1))[None, ...].astype(np.float32)
    return tensor, ratio, pad_w, pad_h


def postprocess_onnx(outputs, ratio, pad_w, pad_h, orig_w, orig_h, conf_thres, nms_thres):
    labels = outputs[0].reshape(-1).astype(np.int64)
    boxes = outputs[1].reshape(-1, 4).astype(np.float32)
    scores = outputs[2].reshape(-1).astype(np.float32)
    preds = []
    for label, box, score in zip(labels, boxes, scores):
        if score < conf_thres:
            continue
        x1 = (box[0] - pad_w) / ratio
        y1 = (box[1] - pad_h) / ratio
        x2 = (box[2] - pad_w) / ratio
        y2 = (box[3] - pad_h) / ratio
        x1 = max(0.0, min(float(x1), float(orig_w - 1)))
        y1 = max(0.0, min(float(y1), float(orig_h - 1)))
        x2 = max(0.0, min(float(x2), float(orig_w)))
        y2 = max(0.0, min(float(y2), float(orig_h)))
        if x2 > x1 and y2 > y1:
            preds.append({"bbox": [x1, y1, x2, y2], "score": float(score), "category_id": int(label)})
    return postprocess_nms_per_class(preds, nms_thres)


def run_onnx_inference(session, image_bgr: np.ndarray, input_size: int, conf: float, nms: float):
    tensor, ratio, pad_w, pad_h = preprocess_bgr(image_bgr, input_size)
    target_sizes = np.array([[input_size, input_size]], dtype=np.int64)
    orig_h, orig_w = image_bgr.shape[:2]
    outputs = session.run(
        ["labels", "boxes", "scores"],
        {"images": tensor, "orig_target_sizes": target_sizes},
    )
    return postprocess_onnx(outputs, ratio, pad_w, pad_h, orig_w, orig_h, conf, nms)


def infer_dataset_onnx(session, dataset: CocoDataset, input_size: int, conf: float, nms: float) -> dict[int, list]:
    preds_by_image = {}
    for img_info in dataset.images:
        path = resolve_image_path(dataset.image_dir, img_info["file_name"])
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise RuntimeError(f"无法读取图片：{path}")
        preds_by_image[img_info["id"]] = run_onnx_inference(session, image_bgr, input_size, conf, nms)
    return preds_by_image


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------


def infer_dataset_hf(model, processor, device, dataset: CocoDataset, conf: float, nms: float, batch_size: int, num_workers: int):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

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

    loader = DataLoader(
        CocoImageListDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    preds_by_image: dict[int, list] = {}
    for images, target_sizes_cpu, image_ids in tqdm(loader, desc=f"{dataset.name}"):
        enc = processor(images=images, return_tensors="pt")
        pixel_values = enc["pixel_values"].to(device)
        target_sizes = target_sizes_cpu.to(device)
        kwargs = {"pixel_values": pixel_values}
        if "pixel_mask" in enc:
            kwargs["pixel_mask"] = enc["pixel_mask"].to(device)
        outputs = model(**kwargs)
        results = processor.post_process_object_detection(outputs, threshold=conf, target_sizes=target_sizes)
        for image_id, res in zip(image_ids, results):
            raw_preds = []
            for s, lab, box in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].cpu().float().numpy()):
                x1, y1, x2, y2 = (float(v) for v in box)
                if x2 <= x1 or y2 <= y1:
                    continue
                raw_preds.append({"bbox": [x1, y1, x2, y2], "score": float(s), "category_id": int(lab)})
            preds_by_image[image_id] = postprocess_nms_per_class(raw_preds, nms)
    return preds_by_image


def evaluate_split(
    preds_by_image: dict[int, list],
    dataset: CocoDataset,
    vis_dir: Path | None,
    vis_ids: list[dict] | None,
) -> dict:
    metrics = compute_map(preds_by_image, dataset)
    print_metrics(f"  {dataset.name}", metrics)
    if vis_dir is not None and vis_ids:
        visualize_by_ids(dataset, preds_by_image, vis_dir, vis_ids)
    return metrics


def eval_model_on_datasets(
    model_path: Path,
    specs: list[DatasetSpec],
    vis_plan: dict[str, dict[str, list[dict]]],
    run_output_dir: Path,
    args,
    conf: float,
) -> dict[str, Any]:
    backend = detect_backend(model_path)
    model_stem = make_safe_name(model_path.stem if backend == "onnx" else model_path.name)
    vis_root = run_output_dir / "vis" / model_stem
    datasets_out: dict[str, dict[str, dict]] = {}

    print(f"\n{'=' * 60}\n模型: {model_path} ({backend})\n{'=' * 60}")

    if backend == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit("缺少 onnxruntime，请先安装：pip install onnxruntime") from exc

        if not model_path.exists():
            raise FileNotFoundError(f"找不到 ONNX 模型：{model_path}")

        providers = resolve_onnx_providers(args.device, args.providers)
        print(f"ONNX providers: {providers}")
        session = ort.InferenceSession(str(model_path), providers=providers)

        for spec in specs:
            if not spec.image_dir.exists():
                raise FileNotFoundError(f"找不到图片目录：{spec.image_dir}")
            print(f"\n--- 数据集: {spec.name} ---")
            ds_metrics: dict[str, dict] = {}
            plan = vis_plan.get(spec.name, {})

            if not args.val_only and spec.train_ann.exists():
                train_ds = load_coco("train", spec.train_ann, spec.image_dir)
                preds = infer_dataset_onnx(session, train_ds, args.input_size, conf, args.nms)
                ds_metrics["train"] = metrics_for_json(
                    evaluate_split(
                        preds, train_ds,
                        vis_root / spec.name / "train",
                        plan.get("train"),
                    )
                )
            elif not args.val_only:
                print(f"警告: 未找到 {spec.name} train 标注，跳过 train。")

            val_ds = load_coco("val", spec.val_ann, spec.image_dir)
            preds = infer_dataset_onnx(session, val_ds, args.input_size, conf, args.nms)
            ds_metrics["val"] = metrics_for_json(
                evaluate_split(preds, val_ds, vis_root / spec.name / "val", plan.get("val"))
            )
            datasets_out[spec.name] = ds_metrics

        return {
            "name": model_stem,
            "path": str(model_path),
            "backend": backend,
            "conf_threshold": conf,
            "onnx_providers": providers,
            "datasets": datasets_out,
        }

    import torch
    from transformers import AutoImageProcessor, Deimv2ForObjectDetection

    if not model_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{model_path}")

    device = resolve_torch_device(args.device)
    print(f"torch device: {device}")
    processor = AutoImageProcessor.from_pretrained(str(model_path))
    model = Deimv2ForObjectDetection.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for spec in specs:
            if not spec.image_dir.exists():
                raise FileNotFoundError(f"找不到图片目录：{spec.image_dir}")
            print(f"\n--- 数据集: {spec.name} ---")
            ds_metrics: dict[str, dict] = {}
            plan = vis_plan.get(spec.name, {})

            if not args.val_only and spec.train_ann.exists():
                train_ds = load_coco("train", spec.train_ann, spec.image_dir)
                preds = infer_dataset_hf(
                    model, processor, device, train_ds, conf, args.nms, args.batch_size, args.num_workers
                )
                ds_metrics["train"] = metrics_for_json(
                    evaluate_split(
                        preds, train_ds,
                        vis_root / spec.name / "train",
                        plan.get("train"),
                    )
                )
            elif not args.val_only:
                print(f"警告: 未找到 {spec.name} train 标注，跳过 train。")

            val_ds = load_coco("val", spec.val_ann, spec.image_dir)
            preds = infer_dataset_hf(
                model, processor, device, val_ds, conf, args.nms, args.batch_size, args.num_workers
            )
            ds_metrics["val"] = metrics_for_json(
                evaluate_split(preds, val_ds, vis_root / spec.name / "val", plan.get("val"))
            )
            datasets_out[spec.name] = ds_metrics

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": model_stem,
        "path": str(model_path),
        "backend": backend,
        "conf_threshold": conf,
        "device": str(device),
        "datasets": datasets_out,
    }


# ---------------------------------------------------------------------------
# 结果校验与图表
# ---------------------------------------------------------------------------


def validate_results(report: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    models = report.get("models", [])
    if not models:
        issues.append("models 列表为空")
    for m in models:
        if "datasets" not in m or not m["datasets"]:
            issues.append(f"模型 {m.get('name')} 无 datasets 指标")
            continue
        for ds_name, splits in m["datasets"].items():
            for split in ("train", "val"):
                if split not in splits:
                    if split == "train" and report.get("eval_params", {}).get("val_only"):
                        continue
                    issues.append(f"{m.get('name')} / {ds_name} 缺少 {split}")
                    continue
                for key in ("AP50", "AP75", "mAP50_95"):
                    if key not in splits[split]:
                        issues.append(f"{m.get('name')} / {ds_name} / {split} 缺少 {key}")
    if issues:
        print("\n[校验警告]")
        for msg in issues:
            print(f"  - {msg}")
    else:
        print("\n[校验] 指标结构完整。")
    return issues


def collect_metric_rows(report: dict[str, Any], metric: MetricName) -> tuple[list[str], list[str], np.ndarray]:
    """返回 (model_names, x_labels, values[model, x])"""
    models = report["models"]
    model_names = [m["name"] for m in models]
    x_labels: list[str] = []
    label_set: list[str] = []
    for m in models:
        for ds_name, splits in m["datasets"].items():
            for split in sorted(splits.keys()):
                label = f"{ds_name}\n{split}"
                if label not in label_set:
                    label_set.append(label)
                    x_labels.append(label)
    values = np.full((len(model_names), len(x_labels)), np.nan, dtype=np.float64)
    col_index = {lb: i for i, lb in enumerate(x_labels)}
    for mi, m in enumerate(models):
        for ds_name, splits in m["datasets"].items():
            for split, sm in splits.items():
                label = f"{ds_name}\n{split}"
                if label in col_index and metric in sm:
                    values[mi, col_index[label]] = float(sm[metric])
    return model_names, x_labels, values


def _draw_heatmap_ax(ax, values: np.ndarray, model_names: list[str], x_labels: list[str], title: str):
    im = ax.imshow(values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    return im


def plot_metric_heatmaps(report: dict[str, Any], charts_dir: Path):
    """一张图两个子图：AP50 与 mAP50-95。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heatmap_metrics: list[tuple[MetricName, str]] = [
        ("AP50", "AP50"),
        ("mAP50_95", "mAP50-95"),
    ]
    panels: list[tuple[str, list[str], list[str], np.ndarray]] = []
    for metric, display in heatmap_metrics:
        model_names, x_labels, values = collect_metric_rows(report, metric)
        if values.size == 0 or np.all(np.isnan(values)):
            print(f"跳过热力图子图 {display}：无有效数据")
            continue
        panels.append((display, model_names, x_labels, values))

    if not panels:
        print("跳过热力图：无有效数据")
        return

    n_cols = len(heatmap_metrics)
    ref_xl = panels[0][2]
    ref_mn = panels[0][1]
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(max(12, len(ref_xl) * 1.4 * n_cols), max(3.5, len(ref_mn) * 0.55)),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    panel_by_metric = {p[0]: p for p in panels}

    for idx, (_metric, display) in enumerate(heatmap_metrics):
        ax = axes_flat[idx]
        if display not in panel_by_metric:
            ax.set_visible(False)
            continue
        _, model_names, x_labels, values = panel_by_metric[display]
        im = _draw_heatmap_ax(ax, values, model_names, x_labels, f"{display} (model × dataset/split)")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    heat_path = charts_dir / "heatmap_AP50_mAP50_95.png"
    fig.savefig(heat_path, dpi=150)
    plt.close(fig)
    print(f"图表已保存: {heat_path}")


def plot_comparison_charts(report: dict[str, Any], charts_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[MetricName] = ["AP50", "AP75", "mAP50_95"]

    for metric in metrics:
        model_names, x_labels, values = collect_metric_rows(report, metric)
        if values.size == 0 or np.all(np.isnan(values)):
            print(f"跳过图表 {metric}：无有效数据")
            continue

        n_models, n_groups = values.shape
        x = np.arange(n_groups)
        width = 0.8 / max(n_models, 1)

        fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.2), 5))
        for i, name in enumerate(model_names):
            offset = (i - (n_models - 1) / 2) * width
            bars = ax.bar(x + offset, values[i], width, label=name)
            for bar, val in zip(bars, values[i]):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=0,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by model × dataset")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, min(1.05, float(np.nanmax(values)) * 1.15 + 0.05) if not np.all(np.isnan(values)) else 1.0)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out_path = charts_dir / f"compare_{metric}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"图表已保存: {out_path}")

    plot_metric_heatmaps(report, charts_dir)


def print_summary_table(report: dict[str, Any]):
    print("\n[指标汇总]")
    header = f"{'模型':<24} {'数据集':<28} {'划分':<6} {'AP50':>8} {'AP75':>8} {'mAP50-95':>10}"
    print(header)
    print("-" * len(header))
    for m in report["models"]:
        for ds_name, splits in m["datasets"].items():
            for split, sm in sorted(splits.items()):
                print(
                    f"{m['name']:<24} {ds_name:<28} {split:<6} "
                    f"{sm['AP50']:>8.4f} {sm['AP75']:>8.4f} {sm['mAP50_95']:>10.4f}"
                )


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    model_paths = list(args.model) if args.model else [Path(p) for p in cfg.DEFAULT_MODELS]
    if not model_paths:
        raise SystemExit("未指定 --model，且 eval_config.DEFAULT_MODELS 为空")

    specs = resolve_dataset_specs()
    vis_plan = build_vis_plan(specs, args.vis_num, args.seed, args.val_only)

    run_name = build_run_name(args, len(model_paths), model_paths)
    run_output_dir = args.output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=False)

    print(f"run_name: {run_name}")
    print(f"输出目录: {run_output_dir}")
    print(f"模型数量: {len(model_paths)}，数据集: {[s.name for s in specs]}")
    print(f"可视化 seed={args.seed}，每划分 {args.vis_num} 张（全模型共用抽样）")

    models_results = []
    for model_path in model_paths:
        backend = detect_backend(model_path)
        conf = args.conf if args.conf is not None else default_conf(backend)
        models_results.append(
            eval_model_on_datasets(model_path, specs, vis_plan, run_output_dir, args, conf)
        )

    report: dict[str, Any] = {
        "run": {
            "name": run_name,
            "output_dir": str(run_output_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
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
            "nms_threshold": args.nms,
            "device": args.device,
            "input_size": args.input_size,
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

    if not args.no_plots:
        charts_dir = run_output_dir / "charts"
        plot_comparison_charts(report, charts_dir)


if __name__ == "__main__":
    main()
