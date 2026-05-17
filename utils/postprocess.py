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


def _postprocess_torch(
    out_logits: torch.Tensor,
    out_bbox: torch.Tensor,
    target_sizes: torch.Tensor,
    *,
    num_classes: int,
    use_focal_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    boxes = center_to_corners_format(out_bbox)
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(
        dtype=boxes.dtype, device=boxes.device
    )
    boxes = boxes * scale_fct[:, None, :]

    num_top_queries = out_logits.shape[1]
    if use_focal_loss:
        scores = torch.sigmoid(out_logits)
        flat = scores.flatten(1)
        scores, index = _topk_deterministic(flat, num_top_queries)
        labels = (index % num_classes).to(torch.int64)
        index = index // num_classes
        boxes = boxes.gather(
            dim=1,
            index=index.unsqueeze(-1).expand(-1, -1, boxes.shape[-1]),
        )
    else:
        scores = torch.softmax(out_logits, dim=-1)[:, :, :-1]
        scores, labels = scores.max(dim=-1)
        if scores.shape[1] > num_top_queries:
            scores, index = _topk_deterministic(scores, num_top_queries)
            labels = torch.gather(labels, dim=1, index=index)
            boxes = torch.gather(
                boxes,
                dim=1,
                index=index.unsqueeze(-1).expand(-1, -1, boxes.shape[-1]),
            )

    return labels, boxes, scores


def _topk_deterministic(flat: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k；并列分数时按列下标稳定破平（便于 ONNX Runtime 与 PyTorch 一致）。"""
    col = torch.arange(flat.shape[1], device=flat.device, dtype=flat.dtype)
    biased = flat - col * 1e-6
    _, index = torch.topk(biased, k, dim=-1)
    scores = torch.gather(flat, 1, index)
    return scores, index
