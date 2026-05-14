"""
ONNX 目标检测模型评测脚本。

功能：
- 按 C++ 调用代码一致的 letterbox 预处理运行 dino_0329_30.onnx
- 对 train / val COCO 标注分别计算 AP50、AP75、mAP@[.50:.95]
- 随机抽图可视化预测框和 ground truth 框

用法：
    python eval_onnx_detector.py
    python eval_onnx_detector.py --conf 0.25 --nms 0.45 --input_size 640
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

try:
    import onnxruntime as ort
except ImportError as exc:
    raise SystemExit("缺少 onnxruntime，请先安装：pip install onnxruntime") from exc

import onnx_eval_config as cfg


@dataclass
class CocoDataset:
    name: str
    ann_path: Path
    image_dir: Path
    images: list[dict]
    gt_by_image: dict[int, list[dict]]
    category_names: dict[int, str]


def parse_args():
    parser = argparse.ArgumentParser(description="ONNX 目标检测模型评测")
    parser.add_argument("--onnx", type=Path, default=cfg.ONNX_MODEL_PATH)
    parser.add_argument("--image_dir", type=Path, default=cfg.IMAGE_DIR)
    parser.add_argument("--train_ann", type=Path, default=cfg.TRAIN_ANN)
    parser.add_argument("--val_ann", type=Path, default=cfg.VAL_ANN)
    parser.add_argument("--output_dir", type=Path, default=cfg.OUTPUT_DIR)
    parser.add_argument("--input_size", type=int, default=cfg.INPUT_SIZE)
    parser.add_argument("--conf", type=float, default=cfg.CONF_THRESHOLD)
    parser.add_argument("--nms", type=float, default=cfg.NMS_THRESHOLD)
    parser.add_argument("--vis_num", type=int, default=cfg.VIS_NUM_IMAGES)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--run_name", type=str, default=None,
                        help="本次评测的输出名称；不填则自动由模型名、数据集名、阈值和时间戳生成")
    parser.add_argument("--providers", nargs="*", default=None,
                        help="onnxruntime providers，例如 CUDAExecutionProvider CPUExecutionProvider")
    return parser.parse_args()


def load_coco(name: str, ann_path: Path, image_dir: Path) -> CocoDataset:
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    category_names = {cat["id"]: cat.get("name", str(cat["id"]))
                      for cat in coco.get("categories", [])}
    gt_by_image = {img["id"]: [] for img in coco["images"]}

    for ann in coco.get("annotations", []):
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0 or ann.get("iscrowd", 0):
            continue
        gt_by_image.setdefault(ann["image_id"], []).append({
            "bbox": [float(x), float(y), float(x + w), float(y + h)],
            "category_id": ann["category_id"],
            "matched": False,
        })

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


def build_run_name(args) -> str:
    if args.run_name:
        return make_safe_name(args.run_name)

    model_name = make_safe_name(args.onnx.stem)
    dataset_name = make_safe_name(args.image_dir.parent.name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{model_name}_{dataset_name}_conf{args.conf:g}_nms{args.nms:g}"


def preprocess_bgr(image_bgr: np.ndarray, input_size: int):
    """严格参考用户给出的 C++ 预处理逻辑。"""
    orig_h, orig_w = image_bgr.shape[:2]
    ratio = min(float(input_size) / orig_w, float(input_size) / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    pad_w = (input_size - new_w) // 2
    pad_h = (input_size - new_h) // 2

    resized = cv2.resize(image_bgr, (new_w, new_h))
    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = canvas.astype(np.float32) / 255.0
    tensor = np.transpose(blob, (2, 0, 1))[None, ...].astype(np.float32)
    return tensor, ratio, pad_w, pad_h


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


def postprocess(outputs, ratio, pad_w, pad_h, orig_w, orig_h, conf_thres, nms_thres):
    """严格参考用户给出的 C++ 反变换逻辑，并增加 NMS。"""
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
            preds.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(score),
                "category_id": int(label),
            })

    if not preds:
        return []

    final_preds = []
    categories = sorted({p["category_id"] for p in preds})
    for cat_id in categories:
        cat_preds = [p for p in preds if p["category_id"] == cat_id]
        cat_boxes = np.array([p["bbox"] for p in cat_preds], dtype=np.float32)
        cat_scores = np.array([p["score"] for p in cat_preds], dtype=np.float32)
        keep = nms_numpy(cat_boxes, cat_scores, nms_thres)
        final_preds.extend(cat_preds[i] for i in keep)

    return sorted(final_preds, key=lambda p: p["score"], reverse=True)


def run_inference(session, image_bgr: np.ndarray, input_size: int, conf: float, nms: float):
    tensor, ratio, pad_w, pad_h = preprocess_bgr(image_bgr, input_size)
    target_sizes = np.array([[input_size, input_size]], dtype=np.int64)
    orig_h, orig_w = image_bgr.shape[:2]

    outputs = session.run(
        ["labels", "boxes", "scores"],
        {"images": tensor, "orig_target_sizes": target_sizes},
    )
    return postprocess(outputs, ratio, pad_w, pad_h, orig_w, orig_h, conf, nms)


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
        cv2.putText(image, f"GT:{name}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)

    for pred in preds:
        x1, y1, x2, y2 = [int(round(v)) for v in pred["bbox"]]
        name = category_names.get(pred["category_id"], str(pred["category_id"]))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.putText(image, f"P:{name} {pred['score']:.2f}", (x1, min(image.shape[0] - 5, y2 + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1, cv2.LINE_AA)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image)


def evaluate_dataset(session, dataset: CocoDataset, input_size: int, conf: float, nms: float,
                     output_dir: Path, vis_num: int, seed: int):
    print(f"\n评测 {dataset.name}: {dataset.ann_path}")
    preds_by_image = {}

    for idx, img_info in enumerate(dataset.images, start=1):
        image_path = resolve_image_path(dataset.image_dir, img_info["file_name"])
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"无法读取图片：{image_path}")

        preds = run_inference(session, image_bgr, input_size, conf, nms)
        preds_by_image[img_info["id"]] = preds

        # if idx % 10 == 0 or idx == len(dataset.images):
        #     print(f"  {idx}/{len(dataset.images)} images done")

    metrics = compute_map(preds_by_image, dataset)
    print(f"{dataset.name} 指标:")
    print(f"  AP50      = {metrics['AP50']:.4f}")
    print(f"  AP75      = {metrics['AP75']:.4f}")
    print(f"  mAP50-95  = {metrics['mAP50_95']:.4f}")

    rng = random.Random(seed)
    samples = rng.sample(dataset.images, k=min(vis_num, len(dataset.images)))
    vis_dir = output_dir / dataset.name

    # print(f"\n{dataset.name} 随机可视化与 GT 打印:")
    for img_info in samples:
        image_path = resolve_image_path(dataset.image_dir, img_info["file_name"])
        image_bgr = cv2.imread(str(image_path))
        preds = preds_by_image.get(img_info["id"], [])
        gts = dataset.gt_by_image.get(img_info["id"], [])

        # print(f"  image_id={img_info['id']} file={img_info['file_name']}")
        # print(f"    GT boxes({len(gts)}):")
        for gt in gts:
            name = dataset.category_names.get(gt["category_id"], str(gt["category_id"]))
            x1, y1, x2, y2 = gt["bbox"]
            # print(f"      {name}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        # print(f"    Pred boxes({len(preds)}):")
        for pred in preds:
            name = dataset.category_names.get(pred["category_id"], str(pred["category_id"]))
            x1, y1, x2, y2 = pred["bbox"]
            # print(f"      {name} score={pred['score']:.3f}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

        save_name = f"{Path(img_info['file_name']).stem}_pred_gt.jpg"
        draw_boxes(image_bgr, preds, gts, dataset.category_names, vis_dir / save_name)

    # print(f"  可视化结果保存到: {vis_dir}")
    return metrics


def main():
    args = parse_args()

    if not args.onnx.exists():
        raise FileNotFoundError(f"找不到 ONNX 模型：{args.onnx}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"找不到图片目录：{args.image_dir}")

    providers = args.providers
    if providers is None:
        available = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

    print(f"ONNX: {args.onnx}")
    print(f"Providers: {providers}")
    print(f"input_size={args.input_size}, conf={args.conf}, nms={args.nms}")

    session = ort.InferenceSession(str(args.onnx), providers=providers)
    train_set = load_coco("train", args.train_ann, args.image_dir)
    val_set = load_coco("val", args.val_ann, args.image_dir)

    run_name = build_run_name(args)
    run_output_dir = args.output_dir / run_name
    run_output_dir.mkdir(parents=True, exist_ok=False)

    print(f"run_name: {run_name}")
    print(f"输出目录: {run_output_dir}")

    train_metrics = evaluate_dataset(
        session, train_set, args.input_size, args.conf, args.nms,
        run_output_dir, args.vis_num, args.seed,
    )
    val_metrics = evaluate_dataset(
        session, val_set, args.input_size, args.conf, args.nms,
        run_output_dir, args.vis_num, args.seed + 1,
    )

    all_metrics = {
        "run": {
            "name": run_name,
            "output_dir": str(run_output_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "model": {
            "name": args.onnx.stem,
            "path": str(args.onnx),
        },
        "dataset": {
            "name": args.image_dir.parent.name,
            "image_dir": str(args.image_dir),
            "train_ann": str(args.train_ann),
            "val_ann": str(args.val_ann),
        },
        "eval_params": {
            "input_size": args.input_size,
            "conf_threshold": args.conf,
            "nms_threshold": args.nms,
            "providers": providers,
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
        },
        # 兼容旧读取方式：保留顶层 train / val。
        "train": train_metrics,
        "val": val_metrics,
    }

    metrics_path = run_output_dir / f"metrics_{run_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n指标已保存: {metrics_path}")


if __name__ == "__main__":
    main()
