"""评测：COCO 数据、mAP、推理、可视化与图表。"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

import configs.eval as cfg
from configs import preprocess as pc
from utils.common import (
    Backend,
    detect_backend,
    display_model_name,
    make_safe_name,
    resolve_torch_device,
    setup_matplotlib_chinese,
)
from utils.onnx import resolve_onnx_providers
from utils.postprocess import postprocess_detections
from utils.preprocess import describe_processor, load_deimv2_processor

MetricName = Literal["AP50", "AP75", "mAP50_95"]

ROOT = Path(__file__).resolve().parent.parent


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


def resolve_dataset_specs() -> list[DatasetSpec]:
    if not cfg.DATASETS:
        raise ValueError("configs/eval.DATASETS 为空，请至少配置一个数据集")
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


# ========================================================================
# Detection Metrics
# ========================================================================

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


def filter_preds_for_vis(preds_by_image: dict[int, list], image_ids: list[int], vis_conf: float) -> dict[int, list]:
    out: dict[int, list] = {}
    for image_id in image_ids:
        preds = preds_by_image.get(image_id, [])
        out[image_id] = [p for p in preds if float(p["score"]) >= vis_conf]
    return out


def preds_for_vis(
    preds_by_image: dict[int, list],
    image_ids: list[int],
    vis_conf: float,
    nms_thres: float,
) -> dict[int, list]:
    """可视化：先 per-class NMS，再按 vis_conf 过滤（mAP 全量推理不做 NMS）。"""
    out: dict[int, list] = {}
    for image_id in image_ids:
        preds = list(preds_by_image.get(image_id, []))
        if nms_thres < 1.0:
            preds = postprocess_nms_per_class(preds, nms_thres)
        out[image_id] = [p for p in preds if float(p["score"]) >= vis_conf]
    return out


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


def compute_map(
    preds_by_image,
    dataset: CocoDataset,
    map_iou_thresholds: list[float],
):
    categories = sorted(dataset.category_names.keys())
    ap_by_iou = {
        thr: compute_ap_for_iou(preds_by_image, dataset.gt_by_image, categories, thr)
        for thr in map_iou_thresholds
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


# ========================================================================
# Eval Vis
# ========================================================================

def vis_split_seed(base_seed: int, dataset_name: str, split: str) -> int:
    h = 0
    for ch in f"{dataset_name}:{split}":
        h = (h * 31 + ord(ch)) & 0x7FFFFFFF
    return (base_seed + h) % (2**31 - 1)


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
        if spec.val_ann.exists():
            val_ds = load_coco("val", spec.val_ann, spec.image_dir)
            infos = pick_vis_image_infos(val_ds, vis_num, vis_split_seed(base_seed, spec.name, "val"))
            plan[spec.name]["val"] = [{"id": int(i["id"]), "file_name": i["file_name"]} for i in infos]
        else:
            print(f"警告: 未找到 {spec.name} val 标注，跳过 val 可视化抽样。")
    return plan

def _vis_draw_scale(height: int, width: int) -> float:
    """大图略放大标注（折中：比固定 0.5/2px 稍大，又不过粗）。"""
    return min(1.35, max(1.0, min(height, width) / 800.0))


def render_boxes_bgr(image_bgr, preds, gts, category_names) -> np.ndarray:
    image = image_bgr.copy()
    h, w = image.shape[:2]
    scale = _vis_draw_scale(h, w)
    box_thickness = max(2, int(round(1.5 * scale)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52 * scale
    text_thickness = 2 if scale >= 1.15 else 1
    line_type = cv2.LINE_AA

    for gt in gts:
        x1, y1, x2, y2 = [int(round(v)) for v in gt["bbox"]]
        name = category_names.get(gt["category_id"], str(gt["category_id"]))
        suffix = " [crowd]" if gt.get("iscrowd") else ""
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), box_thickness)
        label_y = max(int(16 * scale), y1 - int(5 * scale))
        cv2.putText(
            image,
            f"GT:{name}{suffix}",
            (x1, label_y),
            font,
            font_scale,
            (0, 220, 0),
            text_thickness,
            line_type,
        )
    for pred in preds:
        x1, y1, x2, y2 = [int(round(v)) for v in pred["bbox"]]
        name = category_names.get(pred["category_id"], str(pred["category_id"]))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 80, 255), box_thickness)
        label_y = min(h - int(4 * scale), y2 + int(16 * scale))
        cv2.putText(
            image,
            f"P:{name} {pred['score']:.2f}",
            (x1, label_y),
            font,
            font_scale,
            (0, 80, 255),
            text_thickness,
            line_type,
        )
    return image


def _align_panel_width(panel_bgr: np.ndarray, target_w: int) -> np.ndarray:
    h, w = panel_bgr.shape[:2]
    if w == target_w:
        return panel_bgr
    new_h = max(1, int(round(h * target_w / w)))
    return cv2.resize(panel_bgr, (target_w, new_h), interpolation=cv2.INTER_LINEAR)


def _vis_title_bar(width: int, text: str, scale: float) -> np.ndarray:
    bar_h = max(34, int(round(32 * scale)))
    bar = np.full((bar_h, width, 3), 245, dtype=np.uint8)
    cv2.line(bar, (0, bar_h - 1), (width - 1, bar_h - 1), (210, 210, 210), 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1 if scale < 1.15 else 2
    font_scale = 0.52 * scale
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    while tw > width - 20 and font_scale > 0.35:
        font_scale -= 0.05
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    y = (bar_h + th) // 2
    cv2.putText(bar, text, (10, y), font, font_scale, (50, 50, 50), thickness, cv2.LINE_AA)
    return bar


def stack_vis_panels(
    panels: list[tuple[str, np.ndarray, str]],
    *,
    dataset_name: str,
    split: str,
) -> np.ndarray:
    """按原图分辨率纵向拼接多模型面板（每行前加标题条）。"""
    if not panels:
        raise ValueError("stack_vis_panels: panels 为空")
    target_w = max(p[1].shape[1] for p in panels)
    ref_h = panels[0][1].shape[0]
    scale = _vis_draw_scale(ref_h, target_w)
    rows: list[np.ndarray] = []
    for model_stem, panel_bgr, fn in panels:
        panel = _align_panel_width(panel_bgr, target_w)
        title = f"{dataset_name} | {split} | {Path(fn).name} | {model_stem}"
        rows.append(_vis_title_bar(target_w, title, scale))
        rows.append(panel)
    return np.vstack(rows)


def vis_group_dir_name(dataset_name: str, split: str) -> str:
    """可视化分组目录名，例如 teabud_march_ztu_val。"""
    return make_safe_name(f"{dataset_name}_{split}")


@dataclass
class VisPanelAccumulator:
    """内存累积各模型 vis 面板，评测结束后写入 vis/{dataset}_{split}/{image}.jpg。"""

    _panels: dict[tuple[str, str, str], dict[str, tuple[np.ndarray, str]]] = field(
        default_factory=dict
    )

    def add(
        self,
        dataset_name: str,
        split: str,
        image_stem: str,
        file_name: str,
        model_stem: str,
        panel_bgr: np.ndarray,
    ) -> None:
        key = (dataset_name, split, image_stem)
        self._panels.setdefault(key, {})[model_stem] = (panel_bgr, file_name)

    def compose_all(
        self,
        vis_root: Path,
        vis_plan: dict[str, dict[str, list[dict]]],
        model_stems: list[str],
    ) -> int:
        jpeg_q = int(getattr(cfg, "VIS_JPEG_QUALITY", 95))
        jpeg_q = max(1, min(100, jpeg_q))
        n_saved = 0
        for dataset_name, splits in vis_plan.items():
            for split, image_infos in splits.items():
                out_dir = vis_root / vis_group_dir_name(dataset_name, split)
                for info in image_infos:
                    file_name = info["file_name"]
                    image_stem = make_safe_name(Path(file_name).stem)
                    model_map = self._panels.get((dataset_name, split, image_stem), {})
                    panels: list[tuple[str, np.ndarray, str]] = []
                    for model_stem in model_stems:
                        if model_stem not in model_map:
                            continue
                        panel_bgr, fn = model_map[model_stem]
                        panels.append((model_stem, panel_bgr, fn))
                    if not panels:
                        continue

                    stacked = stack_vis_panels(
                        panels, dataset_name=dataset_name, split=split
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{image_stem}.jpg"
                    cv2.imwrite(
                        str(out_path),
                        stacked,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q],
                    )
                    n_saved += 1
        return n_saved


def visualize_by_ids(
    dataset: CocoDataset,
    preds_by_image: dict[int, list],
    vis_accum: VisPanelAccumulator,
    image_infos: list[dict],
    model_stem: str,
    *,
    dataset_name: str,
    split: str,
):
    id_to_info = {int(i["id"]): i for i in image_infos}
    for image_id, img_info in id_to_info.items():
        path = resolve_image_path(dataset.image_dir, img_info["file_name"])
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise RuntimeError(f"无法读取图片：{path}")
        preds = preds_by_image.get(image_id, [])
        gts = dataset.gt_by_image_vis.get(image_id, [])
        file_name = img_info["file_name"]
        image_stem = make_safe_name(Path(file_name).stem)
        panel = render_boxes_bgr(image_bgr, preds, gts, dataset.category_names)
        vis_accum.add(dataset_name, split, image_stem, file_name, model_stem, panel)


def print_metrics(dataset_label: str, metrics: dict):
    print(f"{dataset_label} 指标:")
    print(f"  AP50      = {metrics['AP50']:.4f}")
    print(f"  AP75      = {metrics['AP75']:.4f}")
    print(f"  mAP50-95  = {metrics['mAP50_95']:.4f}")

def build_run_name(args) -> str:
    if args.run_name:
        return make_safe_name(args.run_name)
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_vis_conf_specs(specs: list[str] | None) -> dict[str, float]:
    """解析 ``--vis-conf deimv2_l_march:0.35 dino:0.15``。"""
    if not specs:
        return {}
    out: dict[str, float] = {}
    for item in specs:
        if ":" not in item:
            raise ValueError(
                f"无效 --vis-conf 项 '{item}'，应为 模型键:阈值，例如 deimv2_l_march:0.35"
            )
        key, raw = item.rsplit(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"无效 --vis-conf 项 '{item}'：缺少模型键")
        thr = float(raw.strip())
        if not (0.0 <= thr <= 1.0):
            raise ValueError(f"--vis-conf 阈值须在 [0, 1]：{item}")
        out[key] = thr
    return out


def resolve_vis_conf(
    model_path: Path,
    backend: Backend,
    *,
    default: float,
    config_map: dict[str, float] | None = None,
    cli_map: dict[str, float] | None = None,
) -> float:
    """按路径 / display 名子串匹配可视化 conf；CLI 覆盖 config，最长键优先。"""
    path_s = str(model_path.resolve()).replace("\\", "/")
    display = display_model_name(model_path, backend)
    merged: dict[str, float] = {}
    if config_map:
        merged.update(config_map)
    if cli_map:
        merged.update(cli_map)

    for pattern, conf in sorted(merged.items(), key=lambda kv: len(kv[0]), reverse=True):
        pat = pattern.replace("\\", "/")
        if pat in path_s or pat in display or pat in model_path.name:
            return conf
    return default


# ========================================================================
# Eval Hf
# ========================================================================

def infer_dataset_hf(
    model, processor, device, dataset: CocoDataset, conf: float, batch_size: int, num_workers: int
):
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
            preds_by_image[image_id] = raw_preds
    return preds_by_image


def infer_vis_images_hf(
    model,
    processor,
    device,
    dataset: CocoDataset,
    image_infos: list[dict],
    conf: float,
    nms: float,
) -> dict[int, list]:
    import torch
    from PIL import Image

    preds_by_image: dict[int, list] = {}
    id_to_info = {int(i["id"]): i for i in image_infos}
    for image_id, img_info in id_to_info.items():
        path = resolve_image_path(dataset.image_dir, img_info["file_name"])
        image = Image.open(path).convert("RGB")
        w, h = image.size
        target_sizes = torch.tensor([[h, w]], dtype=torch.int64, device=device)
        enc = processor(images=image, return_tensors="pt")
        pixel_values = enc["pixel_values"].to(device)
        kwargs = {"pixel_values": pixel_values}
        if "pixel_mask" in enc:
            kwargs["pixel_mask"] = enc["pixel_mask"].to(device)
        outputs = model(**kwargs)
        results = processor.post_process_object_detection(outputs, threshold=conf, target_sizes=target_sizes)
        res = results[0]
        raw_preds = []
        for s, lab, box in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].cpu().float().numpy()):
            x1, y1, x2, y2 = (float(v) for v in box)
            if x2 <= x1 or y2 <= y1:
                continue
            raw_preds.append({"bbox": [x1, y1, x2, y2], "score": float(s), "category_id": int(lab)})
        preds_by_image[image_id] = postprocess_nms_per_class(raw_preds, nms)
    return preds_by_image


# ========================================================================
# Eval Onnx
# ========================================================================

@dataclass
class OnnxDeploySpec:
    """ONNX 推理配置：HF+export_onnx 导出，或旧版 letterbox 模型。"""

    session: Any
    mode: Literal["hf_processor", "legacy_letterbox"]
    input_size: int
    processor: Any | None = None
    post_cfg: dict[str, int | bool] | None = None


def load_onnx_deploy_spec(
    model_path: Path,
    device: str | None,
    providers: list[str] | None,
    *,
    config_providers: list[str] | None = None,
) -> OnnxDeploySpec:
    import onnxruntime as ort

    sess_providers = resolve_onnx_providers(device, providers, config_providers=config_providers)
    session = ort.InferenceSession(str(model_path), providers=sess_providers)
    out_names = {o.name for o in session.get_outputs()}
    if "logits" in out_names and "pred_boxes" in out_names:
        meta_path = model_path.with_suffix(".meta.json")
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"HF 风格 ONNX 需要同目录元数据：{meta_path}（由 export_onnx.py 生成）"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ckpt = meta.get("checkpoint")
        if not ckpt or not Path(ckpt).is_dir():
            raise FileNotFoundError(f"meta.json 缺少有效 checkpoint 路径：{meta_path}")

        processor = load_deimv2_processor(ckpt)
        post_cfg = {
            "num_classes": int(meta["num_classes"]),
            "use_focal_loss": bool(meta.get("use_focal_loss", True)),
        }
        return OnnxDeploySpec(
            session=session,
            mode="hf_processor",
            input_size=int(meta.get("input_size", pc.INPUT_SIZE)),
            processor=processor,
            post_cfg=post_cfg,
        )

    return OnnxDeploySpec(
        session=session,
        mode="legacy_letterbox",
        input_size=pc.INPUT_SIZE,
    )


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


def _preds_from_deploy_outputs(labels, boxes, scores, conf_thres: float, nms_thres: float) -> list[dict]:
    labels = np.asarray(labels).reshape(-1)
    boxes = np.asarray(boxes).reshape(-1, 4)
    scores = np.asarray(scores).reshape(-1)
    preds = []
    for label, box, score in zip(labels, boxes, scores):
        if float(score) < conf_thres:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        if x2 <= x1 or y2 <= y1:
            continue
        preds.append(
            {"bbox": [x1, y1, x2, y2], "score": float(score), "category_id": int(label)}
        )
    if nms_thres < 1.0:
        return postprocess_nms_per_class(preds, nms_thres)
    return preds


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
    if nms_thres < 1.0:
        return postprocess_nms_per_class(preds, nms_thres)
    return preds


def run_onnx_inference_legacy(
    session,
    image_bgr: np.ndarray,
    input_size: int,
    conf: float,
    nms: float,
):
    tensor, ratio, pad_w, pad_h = preprocess_bgr(image_bgr, input_size)
    target_sizes = np.array([[input_size, input_size]], dtype=np.int64)
    orig_h, orig_w = image_bgr.shape[:2]
    outputs = session.run(
        ["labels", "boxes", "scores"],
        {"images": tensor, "orig_target_sizes": target_sizes},
    )
    return postprocess_onnx(outputs, ratio, pad_w, pad_h, orig_w, orig_h, conf, nms)


def run_onnx_inference_hf(spec: OnnxDeploySpec, image_bgr: np.ndarray, conf: float, nms: float):
    from PIL import Image

    if spec.processor is None or spec.post_cfg is None:
        raise RuntimeError("OnnxDeploySpec 未配置 processor/post_cfg")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    enc = spec.processor(images=pil, return_tensors="pt")
    pixel_values = enc["pixel_values"].numpy().astype(np.float32)
    orig_h, orig_w = image_bgr.shape[:2]
    target_sizes = np.array([[orig_h, orig_w]], dtype=np.int64)
    logits, pred_boxes = spec.session.run(
        ["logits", "pred_boxes"],
        {"pixel_values": pixel_values},
    )
    labels, boxes, scores = postprocess_detections(
        logits,
        pred_boxes,
        target_sizes,
        num_classes=int(spec.post_cfg["num_classes"]),
        use_focal_loss=bool(spec.post_cfg["use_focal_loss"]),
    )
    return _preds_from_deploy_outputs(labels, boxes, scores, conf, nms)


def run_onnx_inference(spec: OnnxDeploySpec, image_bgr: np.ndarray, conf: float, *, nms_thres: float):
    if spec.mode == "hf_processor":
        return run_onnx_inference_hf(spec, image_bgr, conf, nms_thres)
    return run_onnx_inference_legacy(spec.session, image_bgr, spec.input_size, conf, nms_thres)


def infer_dataset_onnx(spec: OnnxDeploySpec, dataset: CocoDataset, conf: float) -> dict[int, list]:
    """全量 mAP 推理：与 HF 一致，仅 score 阈值，不做 NMS。"""
    preds_by_image = {}
    for img_info in dataset.images:
        path = resolve_image_path(dataset.image_dir, img_info["file_name"])
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise RuntimeError(f"无法读取图片：{path}")
        preds_by_image[img_info["id"]] = run_onnx_inference(spec, image_bgr, conf, nms_thres=1.0)
    return preds_by_image


def infer_vis_images_onnx(
    spec: OnnxDeploySpec, dataset: CocoDataset, image_infos: list[dict], conf: float, nms: float
) -> dict[int, list]:
    preds_by_image: dict[int, list] = {}
    for img_info in image_infos:
        path = resolve_image_path(dataset.image_dir, img_info["file_name"])
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise RuntimeError(f"无法读取图片：{path}")
        preds_by_image[int(img_info["id"])] = run_onnx_inference(
            spec, image_bgr, conf, nms_thres=nms
        )
    return preds_by_image


# ========================================================================
# Eval Runner
# ========================================================================

def evaluate_split(
    preds_by_image: dict[int, list],
    dataset: CocoDataset,
    vis_accum: VisPanelAccumulator | None,
    vis_ids: list[dict] | None,
    model_stem: str,
    *,
    dataset_name: str,
    split: str,
    vis_conf: float,
    nms_thres: float = 1.0,
) -> dict:
    metrics = compute_map(preds_by_image, dataset, cfg.MAP_IOU_THRESHOLDS)
    print_metrics(f"  {dataset.name}", metrics)
    if vis_accum is not None and vis_ids:
        vis_image_ids = [int(i["id"]) for i in vis_ids]
        vis_preds = preds_for_vis(preds_by_image, vis_image_ids, vis_conf, nms_thres)
        visualize_by_ids(
            dataset,
            vis_preds,
            vis_accum,
            vis_ids,
            model_stem,
            dataset_name=dataset_name,
            split=split,
        )
    return metrics

def eval_model_on_datasets(
    model_path: Path,
    specs: list[DatasetSpec],
    vis_plan: dict[str, dict[str, list[dict]]],
    run_output_dir: Path,
    args,
    vis_conf: float,
    map_conf: float,
    vis_accum: VisPanelAccumulator | None = None,
) -> dict[str, Any]:
    backend = detect_backend(model_path)
    model_stem = display_model_name(model_path, backend)
    datasets_out: dict[str, dict[str, dict]] = {}

    print(f"\n{'=' * 60}\n模型: {model_path} ({backend})\n{'=' * 60}")
    if backend == "checkpoint":
        print(
            f"mAP score≥{map_conf:g}（HF 后处理，与 train_tea 一致，无额外 NMS）；"
            f"vis score≥{vis_conf:g}，NMS={args.nms:g}"
        )
    else:
        print(
            f"mAP score≥{map_conf:g}（ONNX 后处理，与 HF 一致，无额外 NMS）；"
            f"vis score≥{vis_conf:g}，NMS={args.nms:g}"
        )

    if backend == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit("缺少 onnxruntime，请先安装：pip install onnxruntime") from exc

        if not model_path.exists():
            raise FileNotFoundError(f"找不到 ONNX 模型：{model_path}")

        onnx_spec = load_onnx_deploy_spec(
            model_path,
            args.device,
            args.providers,
            config_providers=getattr(cfg, "ONNX_PROVIDERS", None),
        )
        print(f"ONNX mode: {onnx_spec.mode}, providers: {onnx_spec.session.get_providers()}")

        for spec in specs:
            if not spec.image_dir.exists():
                raise FileNotFoundError(f"找不到图片目录：{spec.image_dir}")
            print(f"\n--- 数据集: {spec.name} ---")
            ds_metrics: dict[str, dict] = {}
            plan = vis_plan.get(spec.name, {})

            if not args.val_only and spec.train_ann.exists():
                train_ds = load_coco("train", spec.train_ann, spec.image_dir)
                preds = infer_dataset_onnx(onnx_spec, train_ds, map_conf)
                ds_metrics["train"] = metrics_for_json(
                    evaluate_split(
                        preds,
                        train_ds,
                        vis_accum,
                        plan.get("train"),
                        model_stem,
                        dataset_name=spec.name,
                        split="train",
                        vis_conf=vis_conf,
                        nms_thres=1.0,
                    )
                )
            elif not args.val_only:
                print(f"警告: 未找到 {spec.name} train 标注，跳过 train。")

            if spec.val_ann.exists():
                val_ds = load_coco("val", spec.val_ann, spec.image_dir)
                preds = infer_dataset_onnx(onnx_spec, val_ds, map_conf)
                ds_metrics["val"] = metrics_for_json(
                    evaluate_split(
                        preds,
                        val_ds,
                        vis_accum,
                        plan.get("val"),
                        model_stem,
                        dataset_name=spec.name,
                        split="val",
                        vis_conf=vis_conf,
                        nms_thres=1.0,
                    )
                )
            else:
                print(f"警告: 未找到 {spec.name} val 标注，跳过 val。")

            if ds_metrics:
                datasets_out[spec.name] = ds_metrics

        return {
            "name": model_stem,
            "path": str(model_path),
            "backend": backend,
            "map_score_threshold": map_conf,
            "vis_conf_threshold": vis_conf,
            "conf_threshold": vis_conf,
            "onnx_mode": onnx_spec.mode,
            "onnx_providers": onnx_spec.session.get_providers(),
            "datasets": datasets_out,
        }

    import torch
    from transformers import Deimv2ForObjectDetection

    if not model_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{model_path}")

    device = resolve_torch_device(args.device)
    print(f"torch device: {device}")
    processor = load_deimv2_processor(model_path)
    print(f"图像预处理: {describe_processor(processor)}")
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
                    model, processor, device, train_ds, map_conf, args.batch_size, args.num_workers
                )
                ds_metrics["train"] = metrics_for_json(
                    evaluate_split(
                        preds,
                        train_ds,
                        vis_accum,
                        plan.get("train"),
                        model_stem,
                        dataset_name=spec.name,
                        split="train",
                        vis_conf=vis_conf,
                        nms_thres=args.nms,
                    )
                )
            elif not args.val_only:
                print(f"警告: 未找到 {spec.name} train 标注，跳过 train。")

            if spec.val_ann.exists():
                val_ds = load_coco("val", spec.val_ann, spec.image_dir)
                preds = infer_dataset_hf(
                    model, processor, device, val_ds, map_conf, args.batch_size, args.num_workers
                )
                ds_metrics["val"] = metrics_for_json(
                    evaluate_split(
                        preds,
                        val_ds,
                        vis_accum,
                        plan.get("val"),
                        model_stem,
                        dataset_name=spec.name,
                        split="val",
                        vis_conf=vis_conf,
                        nms_thres=args.nms,
                    )
                )
            else:
                print(f"警告: 未找到 {spec.name} val 标注，跳过 val。")

            if ds_metrics:
                datasets_out[spec.name] = ds_metrics

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": model_stem,
        "path": str(model_path),
        "backend": backend,
        "map_score_threshold": map_conf,
        "vis_conf_threshold": vis_conf,
        "conf_threshold": vis_conf,
        "device": str(device),
        "datasets": datasets_out,
    }


def _redraw_vis_splits(
    *,
    specs: list[DatasetSpec],
    vis_plan: dict[str, dict[str, list[dict]]],
    vis_accum: VisPanelAccumulator,
    model_stem: str,
    val_only: bool,
    vis_conf: float,
    infer_split,
) -> None:
    for spec in specs:
        if not spec.image_dir.exists():
            raise FileNotFoundError(f"找不到图片目录：{spec.image_dir}")
        plan = vis_plan.get(spec.name, {})
        print(f"\n--- 数据集: {spec.name} ---")

        if not val_only and (vis_ids := plan.get("train")):
            if spec.train_ann.exists():
                train_ds = load_coco("train", spec.train_ann, spec.image_dir)
                preds = infer_split(train_ds, vis_ids)
                vis_image_ids = [int(i["id"]) for i in vis_ids]
                vis_preds = filter_preds_for_vis(preds, vis_image_ids, vis_conf)
                visualize_by_ids(
                    train_ds,
                    vis_preds,
                    vis_accum,
                    vis_ids,
                    model_stem,
                    dataset_name=spec.name,
                    split="train",
                )
            else:
                print(f"警告: 未找到 {spec.name} train 标注，跳过 train 可视化。")

        if vis_ids := plan.get("val"):
            if spec.val_ann.exists():
                val_ds = load_coco("val", spec.val_ann, spec.image_dir)
                preds = infer_split(val_ds, vis_ids)
                vis_image_ids = [int(i["id"]) for i in vis_ids]
                vis_preds = filter_preds_for_vis(preds, vis_image_ids, vis_conf)
                visualize_by_ids(
                    val_ds,
                    vis_preds,
                    vis_accum,
                    vis_ids,
                    model_stem,
                    dataset_name=spec.name,
                    split="val",
                )
            else:
                print(f"警告: 未找到 {spec.name} val 标注，跳过 val 可视化。")


def redraw_vis_for_model(
    model_path: Path,
    specs: list[DatasetSpec],
    vis_plan: dict[str, dict[str, list[dict]]],
    run_output_dir: Path,
    args,
    vis_conf: float,
    map_conf: float,
    vis_accum: VisPanelAccumulator | None = None,
) -> None:
    """仅对 vis 抽样图重新推理并覆盖 vis/，不重新计算 mAP。"""
    if vis_accum is None:
        return
    backend = detect_backend(model_path)
    model_stem = display_model_name(model_path, backend)

    print(f"\n{'=' * 60}\n[重绘 vis] 模型: {model_path} ({backend})\n{'=' * 60}")
    print(f"vis 重绘 score≥{vis_conf:g}，NMS={args.nms:g}")

    if backend == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit("缺少 onnxruntime，请先安装：pip install onnxruntime") from exc

        if not model_path.exists():
            raise FileNotFoundError(f"找不到 ONNX 模型：{model_path}")

        onnx_spec = load_onnx_deploy_spec(
            model_path,
            args.device,
            args.providers,
            config_providers=getattr(cfg, "ONNX_PROVIDERS", None),
        )
        print(f"ONNX mode: {onnx_spec.mode}, providers: {onnx_spec.session.get_providers()}")

        def infer_split(dataset: CocoDataset, image_infos: list[dict]) -> dict[int, list]:
            return infer_vis_images_onnx(onnx_spec, dataset, image_infos, map_conf, args.nms)

        _redraw_vis_splits(
            specs=specs,
            vis_plan=vis_plan,
            vis_accum=vis_accum,
            model_stem=model_stem,
            val_only=args.val_only,
            vis_conf=vis_conf,
            infer_split=infer_split,
        )
        return

    import torch
    from transformers import Deimv2ForObjectDetection

    if not model_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{model_path}")

    device = resolve_torch_device(args.device)
    print(f"torch device: {device}")
    processor = load_deimv2_processor(model_path)
    model = Deimv2ForObjectDetection.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    try:
        with torch.no_grad():
            def infer_split(dataset: CocoDataset, image_infos: list[dict]) -> dict[int, list]:
                return infer_vis_images_hf(
                    model, processor, device, dataset, image_infos, map_conf, args.nms
                )

            _redraw_vis_splits(
                specs=specs,
                vis_plan=vis_plan,
                vis_accum=vis_accum,
                model_stem=model_stem,
                val_only=args.val_only,
                vis_conf=vis_conf,
                infer_split=infer_split,
            )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def update_model_vis_conf(models_results: list[dict], model_path: Path, vis_conf: float) -> None:
    resolved = normalize_model_path(model_path)
    for m in models_results:
        if normalize_model_path(m.get("path", "")) == resolved:
            m["vis_conf_threshold"] = vis_conf
            m["conf_threshold"] = vis_conf
            return


# ========================================================================
# Eval Report
# ========================================================================

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


def print_summary_table(report: dict[str, Any]) -> None:
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


def refresh_model_display_names(report: dict[str, Any]) -> int:
    """按 models[].path 刷新图例用短名。返回更新条数。"""
    updated = 0
    for m in report.get("models", []):
        path_str = m.get("path")
        if not path_str:
            continue
        p = Path(path_str)
        backend_raw = m.get("backend")
        backend: Backend = (
            backend_raw if backend_raw in ("onnx", "checkpoint") else detect_backend(p)
        )
        new_name = display_model_name(p, backend)
        if m.get("name") != new_name:
            updated += 1
        m["name"] = new_name
    return updated


# ========================================================================
# Eval Metrics Io
# ========================================================================

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolve_metrics_files(targets: list[Path], eval_root: Path) -> list[Path]:
    eval_root = eval_root.expanduser().resolve()
    if not targets:
        files = sorted(eval_root.glob("*/metrics_*.json"))
        if not files:
            raise SystemExit(f"在 {eval_root} 下未找到 */metrics_*.json")
        return files

    out: list[Path] = []
    for raw in targets:
        t = (ROOT / raw).resolve() if not raw.is_absolute() else raw.expanduser().resolve()
        if t.is_file():
            if t.suffix.lower() != ".json":
                raise SystemExit(f"不是 JSON 文件: {t}")
            out.append(t)
            continue
        if not t.is_dir():
            raise SystemExit(f"路径不存在: {t}")
        found = sorted(t.glob("metrics_*.json"))
        if not found:
            raise SystemExit(f"目录中未找到 metrics_*.json: {t}")
        out.extend(found)
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def load_report(metrics_path: Path) -> dict:
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def find_metrics_path(run_dir: Path) -> Path | None:
    """在评测 run 目录下查找 metrics_*.json。"""
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        return None
    found = sorted(run_dir.glob("metrics_*.json"))
    if not found:
        return None
    preferred = run_dir / f"metrics_{run_dir.name}.json"
    if preferred.is_file():
        return preferred
    return found[0]


def normalize_model_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def completed_model_paths(report: dict[str, Any]) -> set[Path]:
    out: set[Path] = set()
    for m in report.get("models", []):
        p = m.get("path")
        if p:
            out.add(normalize_model_path(p))
    return out


def resolve_eval_run(
    args,
    *,
    project_root: Path | None = None,
) -> tuple[str, Path, dict[str, Any] | None, bool]:
    """
    解析评测输出目录。

    续跑（返回 is_resume=True）当：
    - 指定了 ``--resume``（run 目录或 metrics JSON）；
    - ``--output_dir`` 本身已是含 metrics_*.json 的 run 目录；
    - ``--output_dir`` 为 eval 根且 ``--run_name`` 对应子目录已存在 metrics。
    """
    root = project_root or ROOT

    def _resolve(p: Path) -> Path:
        return (root / p).resolve() if not p.is_absolute() else p.expanduser().resolve()

    resume_raw = getattr(args, "resume", None)
    if resume_raw is not None:
        target = _resolve(resume_raw)
        if target.is_file():
            metrics_path = target
            run_dir = target.parent
        elif target.is_dir():
            metrics_path = find_metrics_path(target)
            if metrics_path is None:
                raise SystemExit(f"目录中未找到 metrics_*.json: {target}")
            run_dir = target
        else:
            raise SystemExit(f"--resume 路径不存在: {target}")
        report = load_report(metrics_path)
        run_name = (report.get("run") or {}).get("name") or run_dir.name
        return run_name, run_dir, report, True

    output_dir = _resolve(args.output_dir)

    metrics_here = find_metrics_path(output_dir)
    if metrics_here is not None:
        report = load_report(metrics_here)
        run_name = (report.get("run") or {}).get("name") or output_dir.name
        return run_name, output_dir, report, True

    if args.run_name:
        named_dir = output_dir / make_safe_name(args.run_name)
        metrics_named = find_metrics_path(named_dir)
        if metrics_named is not None:
            report = load_report(metrics_named)
            return make_safe_name(args.run_name), named_dir, report, True

    run_name = build_run_name(args)
    return run_name, output_dir / run_name, None, False


def default_charts_dir(metrics_path: Path, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return metrics_path.parent / "charts"


# ========================================================================
# Eval Charts
# ========================================================================

def _ordered_dataset_names(report: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for entry in report.get("eval_params", {}).get("datasets", []):
        n = entry.get("name")
        if n and n not in names:
            names.append(n)
    for m in report.get("models", []):
        for n in m.get("datasets", {}):
            if n not in names:
                names.append(n)
    return names


def collect_split_metric_matrix(
    report: dict[str, Any],
    metric: MetricName,
    split: str = "val",
) -> tuple[list[str], list[str], np.ndarray]:
    """模型 × 数据集矩阵（指定划分）。"""
    models = report.get("models", [])
    model_names = [m["name"] for m in models]
    dataset_names = _ordered_dataset_names(report)
    values = np.full((len(model_names), len(dataset_names)), np.nan, dtype=np.float64)
    ds_index = {n: i for i, n in enumerate(dataset_names)}
    for mi, m in enumerate(models):
        for ds_name, splits in m.get("datasets", {}).items():
            if ds_name not in ds_index or split not in splits:
                continue
            sm = splits[split]
            if metric in sm:
                values[mi, ds_index[ds_name]] = float(sm[metric])
    return model_names, dataset_names, values


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
    ax.set_yticklabels(model_names, fontsize=8)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    return im


def format_chart_eval_params(report: dict[str, Any]) -> str:
    """评测超参脚注（mAP/vis 阈值、NMS 等），绘制在 charts 图内。"""
    ep = report.get("eval_params") or {}
    parts: list[str] = []
    map_thr = ep.get("map_score_threshold")
    if map_thr is not None:
        parts.append(f"mAP score≥{float(map_thr):g}")
    vis_thr = ep.get("vis_conf_threshold")
    if vis_thr is None:
        vis_thr = ep.get("conf_threshold")
    if vis_thr is not None:
        parts.append(f"vis conf≥{float(vis_thr):g}")
    nms = ep.get("nms_threshold")
    if nms is not None:
        parts.append(f"vis NMS={float(nms):g}")
    isize = ep.get("input_size")
    if isize is not None:
        parts.append(f"input={int(isize)}")
    if ep.get("val_only"):
        parts.append("val_only")
    return "  |  ".join(parts)


def _stamp_chart_eval_params(fig, report: dict[str, Any], *, bottom: float = 0.02) -> None:
    text = format_chart_eval_params(report)
    if text:
        fig.text(0.5, bottom, text, ha="center", va="bottom", fontsize=8, color="#333333")


_BAR_CHART_METRICS: list[tuple[MetricName, str]] = [
    ("AP50", "AP50"),
    ("AP75", "AP75"),
    ("mAP50_95", "mAP50-95"),
]

_HEATMAP_CHART_METRICS: list[tuple[MetricName, str]] = [
    ("AP50", "AP50"),
    ("mAP50_95", "mAP50-95"),
]


def _draw_compare_bars_on_ax(
    ax,
    model_names: list[str],
    x_labels: list[str],
    values: np.ndarray,
    *,
    title: str,
    ylabel: str,
    show_legend: bool,
) -> None:
    n_models, n_groups = values.shape
    x = np.arange(n_groups)
    width = 0.8 / max(n_models, 1)
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
                )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(loc="upper right", fontsize=7)
    ymax = float(np.nanmax(values)) if not np.all(np.isnan(values)) else 1.0
    ax.set_ylim(0, min(1.05, ymax * 1.15 + 0.05))
    ax.grid(axis="y", alpha=0.3)


def plot_combined_comparison_figure(report: dict[str, Any], charts_dir: Path) -> None:
    """compare_AP50 / AP75 / mAP50_95 与 heatmap 竖排合并为一张图。"""
    plt = setup_matplotlib_chinese()

    heat_panels: list[tuple[int, str, list[str], list[str], np.ndarray]] = []
    for col, (metric, display) in enumerate(_HEATMAP_CHART_METRICS):
        model_names, x_labels, values = collect_metric_rows(report, metric)
        if values.size == 0 or np.all(np.isnan(values)):
            continue
        heat_panels.append((col, display, model_names, x_labels, values))

    ref_xl: list[str] = []
    has_bar = False
    for metric, _display in _BAR_CHART_METRICS:
        _mn, xl, vals = collect_metric_rows(report, metric)
        if vals.size and not np.all(np.isnan(vals)):
            has_bar = True
            if not ref_xl:
                ref_xl = xl
    if not ref_xl and heat_panels:
        ref_xl = heat_panels[0][3]

    if not has_bar and not heat_panels:
        print("跳过合并对比图：无有效数据")
        return

    fig_w = max(12, len(ref_xl) * 1.2)
    fig_h = 3.6 * 3 + (4.2 if heat_panels else 0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[1, 1, 1, 1.15],
        hspace=0.42,
        wspace=0.28,
    )

    for row, (metric, display) in enumerate(_BAR_CHART_METRICS):
        model_names, x_labels, values = collect_metric_rows(report, metric)
        ax = fig.add_subplot(gs[row, :])
        if values.size == 0 or np.all(np.isnan(values)):
            ax.set_visible(False)
            continue
        _draw_compare_bars_on_ax(
            ax,
            model_names,
            x_labels,
            values,
            title=f"{display} by model × dataset",
            ylabel=metric,
            show_legend=(row == 0),
        )

    for col, display, model_names, x_labels, values in heat_panels:
        ax = fig.add_subplot(gs[3, col])
        im = _draw_heatmap_ax(
            ax, values, model_names, x_labels, f"{display} (model × dataset/split)"
        )
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout(rect=[0, 0.05, 1, 0.98])
    _stamp_chart_eval_params(fig, report, bottom=0.01)
    out_path = charts_dir / "compare_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"图表已保存: {out_path}")


def plot_metric_heatmaps(report: dict[str, Any], charts_dir: Path):
    """一张图两个子图：AP50 与 mAP50-95。"""
    plt = setup_matplotlib_chinese()

    panels: list[tuple[str, list[str], list[str], np.ndarray]] = []
    for metric, display in _HEATMAP_CHART_METRICS:
        model_names, x_labels, values = collect_metric_rows(report, metric)
        if values.size == 0 or np.all(np.isnan(values)):
            print(f"跳过热力图子图 {display}：无有效数据")
            continue
        panels.append((display, model_names, x_labels, values))

    if not panels:
        print("跳过热力图：无有效数据")
        return

    n_cols = len(_HEATMAP_CHART_METRICS)
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

    for idx, (_metric, display) in enumerate(_HEATMAP_CHART_METRICS):
        ax = axes_flat[idx]
        if display not in panel_by_metric:
            ax.set_visible(False)
            continue
        _, model_names, x_labels, values = panel_by_metric[display]
        im = _draw_heatmap_ax(ax, values, model_names, x_labels, f"{display} (model × dataset/split)")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _stamp_chart_eval_params(fig, report, bottom=0.02)
    heat_path = charts_dir / "heatmap_AP50_mAP50_95.png"
    fig.savefig(heat_path, dpi=150)
    plt.close(fig)
    print(f"图表已保存: {heat_path}")


def plot_comparison_charts(report: dict[str, Any], charts_dir: Path):
    refresh_model_display_names(report)
    plt = setup_matplotlib_chinese()
    charts_dir.mkdir(parents=True, exist_ok=True)

    for metric, display in _BAR_CHART_METRICS:
        model_names, x_labels, values = collect_metric_rows(report, metric)
        if values.size == 0 or np.all(np.isnan(values)):
            print(f"跳过图表 {metric}：无有效数据")
            continue
        fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 1.2), 5))
        _draw_compare_bars_on_ax(
            ax,
            model_names,
            x_labels,
            values,
            title=f"{display} by model × dataset",
            ylabel=metric,
            show_legend=True,
        )
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        _stamp_chart_eval_params(fig, report, bottom=0.02)
        out_path = charts_dir / f"compare_{metric}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"图表已保存: {out_path}")

    plot_metric_heatmaps(report, charts_dir)
    plot_combined_comparison_figure(report, charts_dir)
    plot_comparison_tables(report, charts_dir)



def _matrix_to_cell_text(values: np.ndarray) -> list[list[str]]:
    rows: list[list[str]] = []
    for i in range(values.shape[0]):
        row: list[str] = []
        for j in range(values.shape[1]):
            v = values[i, j]
            row.append(f"{v:.4f}" if not np.isnan(v) else "—")
        rows.append(row)
    return rows


def _style_metric_table(table, values: np.ndarray):
    """按数值深浅着色，便于横向对比（table 含 rowLabels/colLabels 时数据从 (1,1) 起）。"""
    valid = values[~np.isnan(values)]
    vmin = float(valid.min()) if valid.size else 0.0
    vmax = float(valid.max()) if valid.size else 1.0
    span = max(vmax - vmin, 1e-6)
    for (row, col), cell in table.get_celld().items():
        if row <= 0 or col <= 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(weight="bold")
            continue
        ri, ci = row - 1, col - 1
        if ri >= values.shape[0] or ci >= values.shape[1]:
            continue
        v = values[ri, ci]
        if np.isnan(v):
            cell.set_facecolor("#f5f5f5")
            continue
        t = (v - vmin) / span
        cell.set_facecolor((0.92 - 0.35 * t, 0.97, 0.88 - 0.25 * t))
        cell.set_text_props(ha="center")


def save_comparison_table_csv(report: dict[str, Any], path: Path, split: str = "val") -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "split", "AP50", "AP75", "mAP50_95"])
        for m in report.get("models", []):
            for ds_name, splits in m.get("datasets", {}).items():
                if split not in splits:
                    continue
                sm = splits[split]
                w.writerow(
                    [
                        m["name"],
                        ds_name,
                        split,
                        f"{sm.get('AP50', float('nan')):.6f}",
                        f"{sm.get('AP75', float('nan')):.6f}",
                        f"{sm.get('mAP50_95', float('nan')):.6f}",
                    ]
                )


def save_comparison_table_markdown(
    report: dict[str, Any], path: Path, split: str = "val", metric: MetricName = "mAP50_95"
) -> None:
    model_names, dataset_names, values = collect_split_metric_matrix(report, metric, split=split)
    if values.size == 0:
        return
    display = {"AP50": "AP50", "AP75": "AP75", "mAP50_95": "mAP50-95"}[metric]
    lines = [
        f"# 验证集对比表 ({split}) — {display}",
        "",
        "| 模型 | " + " | ".join(dataset_names) + " |",
        "| --- | " + " | ".join(["---:"] * len(dataset_names)) + " |",
    ]
    cells = _matrix_to_cell_text(values)
    for name, row in zip(model_names, cells):
        lines.append("| " + name + " | " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_comparison_tables(report: dict[str, Any], charts_dir: Path, split: str = "val") -> None:
    """生成模型×数据集对比表（PNG + CSV + Markdown）。"""
    plt = setup_matplotlib_chinese()

    table_metrics: list[tuple[MetricName, str]] = [
        ("AP50", "AP50"),
        ("AP75", "AP75"),
        ("mAP50_95", "mAP50-95"),
    ]
    panels: list[tuple[str, list[str], list[str], np.ndarray]] = []
    for metric, display in table_metrics:
        model_names, dataset_names, values = collect_split_metric_matrix(report, metric, split=split)
        if values.size == 0 or np.all(np.isnan(values)):
            print(f"跳过对比表子图 {display}：无有效 {split} 数据")
            continue
        panels.append((display, model_names, dataset_names, values))

    if not panels:
        print(f"跳过对比表：无有效 {split} 数据")
        return

    charts_dir.mkdir(parents=True, exist_ok=True)
    save_comparison_table_csv(report, charts_dir / f"comparison_table_{split}.csv", split=split)
    save_comparison_table_markdown(
        report, charts_dir / f"comparison_table_{split}_mAP.md", split=split, metric="mAP50_95"
    )
    print(f"对比表 CSV: {charts_dir / f'comparison_table_{split}.csv'}")
    print(f"对比表 Markdown: {charts_dir / f'comparison_table_{split}_mAP.md'}")

    n_rows = len(panels)
    n_cols = len(panels[0][2])
    fig_h = max(2.2 * n_rows + 1.0, 4.0)
    fig_w = max(2.0 + n_cols * 1.35, 8.0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), squeeze=False)

    split_label = {"val": "验证集", "train": "训练集"}.get(split, split)
    for ax_row, (display, model_names, dataset_names, values) in zip(axes.ravel(), panels):
        ax_row.axis("off")
        cell_text = _matrix_to_cell_text(values)
        table = ax_row.table(
            cellText=cell_text,
            rowLabels=model_names,
            colLabels=dataset_names,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.35)
        _style_metric_table(table, values)
        ax_row.set_title(f"{display}", fontsize=11, pad=12)

    fig.suptitle(f"模型 × 数据集 — {split_label} ({split})", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    _stamp_chart_eval_params(fig, report, bottom=0.01)
    out_path = charts_dir / f"comparison_table_{split}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"对比表已保存: {out_path}")


def regenerate_charts_from_report(report: dict[str, Any], charts_dir: Path) -> None:
    """从 report 重绘 charts/ 下全部对比图与表（会先刷新模型显示名）。"""
    n = refresh_model_display_names(report)
    if n:
        print(f"已刷新 {n} 个模型显示名（用于图例）")
    validate_results(report)
    print_summary_table(report)
    charts_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison_charts(report, charts_dir)