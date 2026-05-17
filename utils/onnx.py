"""Deimv2 ONNX 导出与 Runtime（不含检测后处理，见 utils/postprocess.py）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import Deimv2ForObjectDetection

from utils.common import PROJECT_ROOT
from utils.postprocess import postprocess_detections
from utils.preprocess import preprocess_settings_dict


class Deimv2OnnxCore(nn.Module):
    """仅导出 Deimv2 前向（logits + pred_boxes）。"""

    def __init__(self, model: Deimv2ForObjectDetection):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits, outputs.pred_boxes


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
    return PROJECT_ROOT / "onnx_models" / f"{checkpoint_dir.parent.name}_{checkpoint_dir.name}.onnx"


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
) -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("验证需要 onnxruntime：pip install onnxruntime") from exc

    from PIL import Image

    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (600, 800, 3), dtype=np.uint8))
    enc = processor(images=image, return_tensors="pt")
    pixel_values = enc["pixel_values"]
    h, w = image.size[1], image.size[0]
    target_sizes = torch.tensor([[h, w]], dtype=torch.int64)

    verify_device = torch.device("cpu")
    model_cpu = model.to(verify_device).eval()
    pixel_values = pixel_values.to(verify_device)
    target_sizes = target_sizes.to(verify_device)

    with torch.no_grad():
        out = model_cpu(pixel_values=pixel_values)
        pt_logits, pt_pb = out.logits, out.pred_boxes

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_logits, ort_pb = session.run(
        ["logits", "pred_boxes"],
        {"pixel_values": pixel_values.numpy()},
    )

    if not np.allclose(pt_logits.numpy(), ort_logits, rtol=1e-5, atol=2e-4):
        diff = float(np.abs(pt_logits.numpy() - ort_logits).max())
        raise RuntimeError(f"验证失败：logits 最大差异 {diff:.6g}")
    print(f"  logits: ok (shape {pt_logits.shape})")

    if not np.allclose(pt_pb.numpy(), ort_pb, rtol=1e-5, atol=2e-4):
        diff = float(np.abs(pt_pb.numpy() - ort_pb).max())
        raise RuntimeError(f"验证失败：pred_boxes 最大差异 {diff:.6g}")
    print(f"  pred_boxes: ok (shape {pt_pb.shape})")

    labels_o, boxes_o, scores_o = postprocess_detections(
        ort_logits,
        ort_pb,
        target_sizes.numpy(),
        num_classes=int(post_cfg["num_classes"]),
        use_focal_loss=bool(post_cfg["use_focal_loss"]),
    )
    if labels_o.size == 0 or boxes_o.size == 0 or scores_o.size == 0:
        raise RuntimeError("验证失败：后处理输出为空")
    print(
        f"  postprocess: ok（{labels_o.shape}，scores∈[{scores_o.min():.4f}, {scores_o.max():.4f}]）"
    )
    print("ONNX 与 PyTorch 一致（图内 logits/pred_boxes；后处理在 Python 与 eval 相同）。")
