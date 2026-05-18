"""Deimv2 / RT-DETR 风格检测后处理（PyTorch / ONNX 推理共用）。"""

from __future__ import annotations

import numpy as np
import torch
from transformers import Deimv2ForObjectDetection
from transformers.image_transforms import center_to_corners_format


def postprocess_config_from_model(model: Deimv2ForObjectDetection) -> dict[str, int | bool]:
    cfg = model.config
    return {
        "num_queries": int(cfg.num_queries),
        "num_classes": len(cfg.id2label),
        "use_focal_loss": bool(getattr(cfg, "use_focal_loss", True)),
    }


def postprocess_detections(
    out_logits: torch.Tensor | np.ndarray,
    out_bbox: torch.Tensor | np.ndarray,
    target_sizes: torch.Tensor | np.ndarray,
    *,
    num_classes: int,
    use_focal_loss: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RT-DETR 风格后处理（与 HF Deimv2 部署逻辑一致），返回 numpy。"""
    if not isinstance(out_logits, torch.Tensor):
        out_logits = torch.as_tensor(out_logits)
    if not isinstance(out_bbox, torch.Tensor):
        out_bbox = torch.as_tensor(out_bbox)
    if not isinstance(target_sizes, torch.Tensor):
        target_sizes = torch.as_tensor(target_sizes)

    labels, boxes, scores = _postprocess_torch(
        out_logits,
        out_bbox,
        target_sizes,
        num_classes=num_classes,
        use_focal_loss=use_focal_loss,
    )
    return labels.cpu().numpy(), boxes.cpu().numpy(), scores.cpu().numpy()


def _scale_boxes_to_image(
    out_bbox: torch.Tensor,
    target_sizes: torch.Tensor,
) -> torch.Tensor:
    boxes = center_to_corners_format(out_bbox)
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(
        dtype=boxes.dtype, device=boxes.device
    )
    return boxes * scale_fct[:, None, :]


def _gather_topk_boxes(
    scaled_boxes: torch.Tensor,
    query_index: torch.Tensor,
) -> torch.Tensor:
    return scaled_boxes.gather(
        dim=1,
        index=query_index.unsqueeze(-1).expand(-1, -1, scaled_boxes.shape[-1]),
    )


def _topk_from_logits(
    out_logits: torch.Tensor,
    *,
    num_classes: int,
    use_focal_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 top-k 的 scores、labels、query_index（用于 gather pred_boxes）。"""
    num_top_queries = out_logits.shape[1]
    if use_focal_loss:
        scores = torch.sigmoid(out_logits)
        flat = scores.flatten(1)
        scores, flat_index = _topk_deterministic(flat, num_top_queries)
        labels = (flat_index % num_classes).to(torch.int64)
        query_index = flat_index // num_classes
        return scores, labels, query_index

    scores = torch.softmax(out_logits, dim=-1)[:, :, :-1]
    scores, labels = scores.max(dim=-1)
    query_index = torch.arange(
        labels.shape[1], device=labels.device, dtype=torch.int64
    ).expand(labels.shape[0], -1)
    if scores.shape[1] > num_top_queries:
        scores, query_index = _topk_deterministic(scores, num_top_queries)
        labels = torch.gather(labels, dim=1, index=query_index)
    return scores, labels, query_index


def postprocess_boxes_with_shared_topk(
    logits_ref: torch.Tensor | np.ndarray,
    pred_boxes_a: torch.Tensor | np.ndarray,
    pred_boxes_b: torch.Tensor | np.ndarray,
    target_sizes: torch.Tensor | np.ndarray,
    *,
    num_classes: int,
    use_focal_loss: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """用 logits_ref 的 top-k query 索引分别 gather 两组 pred_boxes（用于 ONNX 验证）。"""
    if not isinstance(logits_ref, torch.Tensor):
        logits_ref = torch.as_tensor(logits_ref)
    if not isinstance(pred_boxes_a, torch.Tensor):
        pred_boxes_a = torch.as_tensor(pred_boxes_a)
    if not isinstance(pred_boxes_b, torch.Tensor):
        pred_boxes_b = torch.as_tensor(pred_boxes_b)
    if not isinstance(target_sizes, torch.Tensor):
        target_sizes = torch.as_tensor(target_sizes)

    _, _, query_index = _topk_from_logits(
        logits_ref,
        num_classes=num_classes,
        use_focal_loss=use_focal_loss,
    )
    boxes_a = _gather_topk_boxes(_scale_boxes_to_image(pred_boxes_a, target_sizes), query_index)
    boxes_b = _gather_topk_boxes(_scale_boxes_to_image(pred_boxes_b, target_sizes), query_index)
    return boxes_a.cpu().numpy(), boxes_b.cpu().numpy()


def _postprocess_torch(
    out_logits: torch.Tensor,
    out_bbox: torch.Tensor,
    target_sizes: torch.Tensor,
    *,
    num_classes: int,
    use_focal_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores, labels, query_index = _topk_from_logits(
        out_logits,
        num_classes=num_classes,
        use_focal_loss=use_focal_loss,
    )
    boxes = _gather_topk_boxes(_scale_boxes_to_image(out_bbox, target_sizes), query_index)
    return labels, boxes, scores


def _topk_deterministic(flat: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k；并列分数时按列下标稳定破平（便于 ONNX Runtime 与 PyTorch 一致）。"""
    col = torch.arange(flat.shape[1], device=flat.device, dtype=flat.dtype)
    biased = flat - col * 1e-6
    _, index = torch.topk(biased, k, dim=-1)
    scores = torch.gather(flat, 1, index)
    return scores, index
