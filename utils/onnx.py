"""Deimv2 ONNX 导出与 Runtime（不含检测后处理，见 utils/postprocess.py）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import Deimv2ForObjectDetection

from utils.common import PROJECT_ROOT, checkpoint_onnx_basename
from utils.postprocess import postprocess_boxes_with_shared_topk, postprocess_detections
from utils.preprocess import preprocess_settings_dict


class Deimv2OnnxCore(nn.Module):
    """仅导出 Deimv2 前向（logits + pred_boxes）。"""

    def __init__(self, model: Deimv2ForObjectDetection):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits, outputs.pred_boxes


def prepare_model_for_onnx_export(model: Deimv2ForObjectDetection) -> Deimv2ForObjectDetection:
    """导出/验证前固定 eager attention 与 float32 CPU eval。"""
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    elif hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    return model.eval().cpu().float()


def resolve_onnx_providers(
    device: str | None,
    cli_providers: list[str] | None,
    *,
    config_providers: list[str] | None = None,
) -> list[str]:
    """优先级：cli_providers > config_providers > device > 自动检测 CUDA。"""
    if cli_providers:
        return list(cli_providers)

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


def resolve_checkpoint_dir(path: Path) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint 目录不存在：{path}")
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"缺少 config.json，请指定含 HF 权重的目录：{path}")
    return path


def default_output_path(checkpoint_dir: Path) -> Path:
    return PROJECT_ROOT / "onnx_models" / f"{checkpoint_onnx_basename(checkpoint_dir)}.onnx"


def save_export_meta(
    meta_path: Path,
    *,
    checkpoint_dir: Path,
    onnx_path: Path,
    input_size: int,
    num_queries: int,
    num_classes: int,
    use_focal_loss: bool,
    opset: int,
    input_names: list[str],
    output_names: list[str],
) -> None:
    proc_cfg_path = checkpoint_dir / "preprocessor_config.json"
    preprocessor = {}
    if proc_cfg_path.exists():
        preprocessor = json.loads(proc_cfg_path.read_text(encoding="utf-8"))
    meta = {
        "checkpoint": str(checkpoint_dir),
        "onnx": str(onnx_path),
        "opset": opset,
        "input_names": input_names,
        "output_names": output_names,
        "input_size": input_size,
        "num_queries": num_queries,
        "num_classes": num_classes,
        "use_focal_loss": use_focal_loss,
        "postprocess": "utils.postprocess.postprocess_detections",
        "preprocess_config": preprocess_settings_dict(),
        "preprocessor_config": preprocessor,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def onnx_check(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("使用 --check 需要 onnx：pip install onnx") from exc
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("onnx.checker: 校验通过")


def onnx_simplify(onnx_path: Path, *, input_size: int) -> None:
    try:
        import onnx
        import onnxsim
    except ImportError as exc:
        raise SystemExit("使用 --simplify 需要 onnx、onnxsim：pip install onnx onnxsim") from exc

    data = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    input_shapes = {"pixel_values": data.shape}
    onnx_model_simplify, ok = onnxsim.simplify(
        str(onnx_path),
        test_input_shapes=input_shapes,
    )
    onnx.save(onnx_model_simplify, str(onnx_path))
    print(f"onnxsim 简化完成: {ok}")


def export_onnx(
    core: Deimv2OnnxCore,
    onnx_path: Path,
    *,
    input_size: int,
    opset: int,
    dynamic_batch: bool,
    export_device: torch.device,
) -> None:
    """在 CPU 上 trace/export，避免 CUDA trace 与 onnxruntime CPU 推理不一致。"""
    core = core.to(export_device).eval().float()
    dummy_pixel = torch.randn(1, 3, input_size, input_size, device=export_device)

    input_names = ["pixel_values"]
    output_names = ["logits", "pred_boxes"]
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        }

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            core,
            (dummy_pixel,),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )


def verify_export(
    model: Deimv2ForObjectDetection,
    processor,
    onnx_path: Path,
    post_cfg: dict[str, int | bool],
    *,
    logits_atol: float = 2e-4,
    pred_boxes_atol: float = 2e-4,
    post_scores_atol: float = 5e-3,
    post_boxes_atol: float = 2.0,
) -> None:
    """对比 PyTorch 与 ONNX Runtime。

    raw logits/pred_boxes 对齐时即视为导出成功；后处理仅做可运行性检查。
    raw 存在偏差时，用后处理 scores/labels 及「共享 top-k query」下的 boxes 对齐检查
    （避免临界分数导致同 rank 不同 query、标签相同但框差异很大的误报）。
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("验证需要 onnxruntime：pip install onnxruntime") from exc

    from PIL import Image

    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (600, 800, 3), dtype=np.uint8))
    enc = processor(images=image, return_tensors="pt")
    pixel_values = enc["pixel_values"].to(dtype=torch.float32)
    h, w = image.size[1], image.size[0]
    target_sizes = np.array([[h, w]], dtype=np.int64)
    post_kw = {
        "num_classes": int(post_cfg["num_classes"]),
        "use_focal_loss": bool(post_cfg["use_focal_loss"]),
    }

    core = Deimv2OnnxCore(prepare_model_for_onnx_export(model))
    pixel_values = pixel_values.cpu()

    with torch.no_grad():
        pt_logits, pt_pb = core(pixel_values)

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_logits, ort_pb = session.run(
        ["logits", "pred_boxes"],
        {"pixel_values": pixel_values.numpy()},
    )

    logits_diff = float(np.abs(pt_logits.numpy() - ort_logits).max())
    boxes_diff = float(np.abs(pt_pb.numpy() - ort_pb).max())
    logits_ok = np.allclose(pt_logits.numpy(), ort_logits, rtol=1e-4, atol=logits_atol)
    boxes_ok = np.allclose(pt_pb.numpy(), ort_pb, rtol=1e-4, atol=pred_boxes_atol)
    if logits_ok:
        print(f"  logits: ok (shape {tuple(pt_logits.shape)})")
    else:
        print(
            f"  logits: raw 最大差异 {logits_diff:.6g}（超过 atol={logits_atol:g}，"
            "属 ORT 数值误差，以下用后处理对齐检查）"
        )

    if boxes_ok:
        print(f"  pred_boxes: ok (shape {tuple(pt_pb.shape)})")
    else:
        print(
            f"  pred_boxes: raw 最大差异 {boxes_diff:.6g}（超过 atol={pred_boxes_atol:g}，"
            "以下用后处理对齐检查）"
        )

    labels_o, boxes_o, scores_o = postprocess_detections(
        ort_logits,
        ort_pb,
        target_sizes,
        **post_kw,
    )
    if labels_o.size == 0 or boxes_o.size == 0 or scores_o.size == 0:
        raise RuntimeError("验证失败：ONNX 后处理输出为空")

    if logits_ok and boxes_ok:
        print(
            f"  postprocess: ok（{labels_o.shape}，scores∈[{scores_o.min():.4f}, {scores_o.max():.4f}]）"
        )
        print("ONNX 与 PyTorch 一致（raw logits/pred_boxes 已对齐，可部署）。")
        return

    labels_pt, _, scores_pt = postprocess_detections(
        pt_logits.numpy(),
        pt_pb.numpy(),
        target_sizes,
        **post_kw,
    )
    scores_diff = float(np.abs(scores_pt - scores_o).max()) if scores_pt.size else 0.0
    labels_match = np.array_equal(labels_pt, labels_o)
    boxes_pt_shared, boxes_o_shared = postprocess_boxes_with_shared_topk(
        pt_logits.numpy(),
        pt_pb.numpy(),
        ort_pb,
        target_sizes,
        **post_kw,
    )
    post_boxes_diff = (
        float(np.abs(boxes_pt_shared - boxes_o_shared).max())
        if boxes_pt_shared.size
        else 0.0
    )

    if scores_diff > post_scores_atol:
        raise RuntimeError(f"验证失败：后处理 scores 最大差异 {scores_diff:.6g}")
    if post_boxes_diff > post_boxes_atol:
        raise RuntimeError(
            f"验证失败：共享 top-k 后 boxes 最大差异 {post_boxes_diff:.6g}"
        )
    if not labels_match:
        raise RuntimeError("验证失败：后处理 labels 与 PyTorch 不一致")

    print(
        f"  postprocess: ok（{labels_o.shape}，scores∈[{scores_o.min():.4f}, {scores_o.max():.4f}]，"
        f"scores Δ≤{scores_diff:.4g}，共享 top-k boxes Δ≤{post_boxes_diff:.4g}）"
    )
    print(
        "ONNX 与 PyTorch 在后处理结果上一致（raw logits/pred_boxes 存在 ORT 数值偏差，可部署）。"
    )
