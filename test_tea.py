"""
茶叶目标检测推理 + 可视化（无 mAP 评测）。

用法：
    python test_tea.py --input images/foo.png
    python test_tea.py --input images/
    python test_tea.py --model outputs/deimv2_l_march/checkpoint-best --input images/
    python test_tea.py --model onnx_models/deimv2_l_march.onnx --conf 0.25 --input images/

输入为单张图片或文件夹（仅处理文件夹一级目录下的图片）；结果保存在输入路径下的 vis/ 目录（JPEG）。
多模型时同一张图纵向拼接对比。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import configs.eval as cfg
from configs import preprocess as pc
from utils.common import detect_backend, display_model_name, make_safe_name, resolve_torch_device
from utils.eval import (
    OnnxDeploySpec,
    load_onnx_deploy_spec,
    parse_vis_conf_specs,
    postprocess_nms_per_class,
    render_boxes_bgr,
    resolve_vis_conf,
    run_onnx_inference,
    stack_vis_panels,
)
from utils.preprocess import add_preprocess_arguments, apply_preprocess_from_namespace, load_deimv2_processor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="茶叶检测推理 + 可视化")
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="图片路径或包含图片的文件夹",
    )
    p.add_argument(
        "input_pos",
        nargs="?",
        type=Path,
        default=None,
        help="同 --input（位置参数写法）",
    )
    p.add_argument(
        "--model",
        nargs="*",
        type=Path,
        default=None,
        help="一个或多个模型路径；默认 configs/eval.MODELS 的第一项",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=None,
        help=f"检测 score 下限；默认 {cfg.CONF_THRESHOLD}",
    )
    p.add_argument(
        "--vis-conf",
        nargs="*",
        default=None,
        metavar="KEY:THR",
        help="按模型覆盖 score 下限，例如 deimv2_l_march:0.35",
    )
    p.add_argument("--nms", type=float, default=cfg.NMS_THRESHOLD)
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
    add_preprocess_arguments(p)
    return p.parse_args(argv)


def collect_image_paths(input_path: Path) -> tuple[Path, list[Path]]:
    path = input_path.resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"不支持的图片格式：{path.suffix}（支持 {sorted(IMAGE_EXTENSIONS)}）")
        return path.parent, [path]
    if path.is_dir():
        images = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"文件夹内未找到图片：{path}")
        return path, images
    raise FileNotFoundError(f"找不到输入路径：{input_path}")


def load_category_names_from_config(config_path: Path) -> dict[int, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id2label = data.get("id2label") or {}
    if id2label:
        return {int(k): str(v) for k, v in id2label.items()}
    num_labels = int(data.get("num_labels") or data.get("num_classes") or 1)
    return {i: str(i) for i in range(num_labels)}


def load_category_names(model_path: Path, backend: str, onnx_spec: OnnxDeploySpec | None) -> dict[int, str]:
    if backend == "checkpoint":
        return load_category_names_from_config(model_path / "config.json")
    if onnx_spec is not None and onnx_spec.mode == "hf_processor":
        meta_path = model_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ckpt = Path(meta["checkpoint"])
        return load_category_names_from_config(ckpt / "config.json")
    num_classes = int(onnx_spec.post_cfg["num_classes"]) if onnx_spec and onnx_spec.post_cfg else 1
    return {i: str(i) for i in range(num_classes)}


def infer_image_hf(model, processor, device, image_path: Path, conf: float, nms: float) -> list[dict]:
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
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
    preds = []
    for s, lab, box in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].cpu().float().numpy()):
        x1, y1, x2, y2 = (float(v) for v in box)
        if x2 <= x1 or y2 <= y1:
            continue
        preds.append({"bbox": [x1, y1, x2, y2], "score": float(s), "category_id": int(lab)})
    return postprocess_nms_per_class(preds, nms)


class HfRunner:
    def __init__(self, model_path: Path, device_arg: str | None):
        import torch
        from transformers import Deimv2ForObjectDetection

        if not model_path.exists():
            raise FileNotFoundError(f"找不到 checkpoint：{model_path}")
        self.model_path = model_path
        self.device = resolve_torch_device(device_arg)
        self.processor = load_deimv2_processor(model_path)
        self.model = Deimv2ForObjectDetection.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        self.category_names = load_category_names(model_path, "checkpoint", None)
        print(f"HF checkpoint: {model_path.name}, device={self.device}")

    def infer(self, image_path: Path, conf: float, nms: float) -> list[dict]:
        import torch

        with torch.no_grad():
            return infer_image_hf(self.model, self.processor, self.device, image_path, conf, nms)

    def close(self) -> None:
        import torch

        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class OnnxRunner:
    def __init__(self, model_path: Path, device_arg: str | None, providers: list[str] | None):
        if not model_path.exists():
            raise FileNotFoundError(f"找不到 ONNX 模型：{model_path}")
        self.model_path = model_path
        self.spec = load_onnx_deploy_spec(
            model_path,
            device_arg,
            providers,
            config_providers=getattr(cfg, "ONNX_PROVIDERS", None),
        )
        self.category_names = load_category_names(model_path, "onnx", self.spec)
        print(f"ONNX: {model_path.name}, mode={self.spec.mode}, providers={self.spec.session.get_providers()}")

    def infer(self, image_path: Path, conf: float, nms: float) -> list[dict]:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"无法读取图片：{image_path}")
        return run_onnx_inference(self.spec, image_bgr, conf, nms_thres=nms)


def build_runners(model_paths: list[Path], args) -> list[tuple[Path, str, HfRunner | OnnxRunner]]:
    runners: list[tuple[Path, str, HfRunner | OnnxRunner]] = []
    for model_path in model_paths:
        backend = detect_backend(model_path)
        stem = display_model_name(model_path, backend)
        if backend == "onnx":
            runners.append((model_path, stem, OnnxRunner(model_path, args.device, args.providers)))
        else:
            runners.append((model_path, stem, HfRunner(model_path, args.device)))
    return runners


def save_vis(
    panels: list[tuple[str, np.ndarray, str]],
    out_path: Path,
    *,
    source_name: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jpeg_q = max(1, min(100, int(getattr(cfg, "VIS_JPEG_QUALITY", 95))))
    if len(panels) == 1:
        cv2.imwrite(str(out_path), panels[0][1], [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
        return
    stacked = stack_vis_panels(panels, dataset_name="test", split=source_name)
    cv2.imwrite(str(out_path), stacked, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    input_path = args.input or args.input_pos
    if input_path is None:
        raise SystemExit("请指定输入路径：--input <路径> 或位置参数")
    prep_applied = apply_preprocess_from_namespace(args)
    if prep_applied:
        print("预处理 CLI 覆盖: " + ", ".join(prep_applied))

    model_paths = list(args.model) if args.model else [Path(p) for p in cfg.MODELS[:1]]
    if not model_paths:
        raise SystemExit("未指定 --model，且 configs/eval.MODELS 为空")

    output_root, image_paths = collect_image_paths(input_path)
    vis_dir = output_root / "vis"
    conf_default = args.conf if args.conf is not None else cfg.CONF_THRESHOLD
    cli_vis_conf = parse_vis_conf_specs(args.vis_conf)
    config_vis_conf = dict(getattr(cfg, "VIS_CONF_BY_MODEL", {}) or {})

    print(f"输入: {input_path.resolve()}")
    print(f"图片数量: {len(image_paths)}")
    print(f"输出目录: {vis_dir}")
    print(f"模型数量: {len(model_paths)}，conf 默认={conf_default:g}，NMS={args.nms:g}")
    print(f"input_size={pc.INPUT_SIZE}")

    runners = build_runners(model_paths, args)
    try:
        for image_path in image_paths:
            rel_name = image_path.name
            if image_path.is_relative_to(output_root):
                rel_name = str(image_path.relative_to(output_root))
            image_stem = make_safe_name(Path(rel_name).with_suffix("").as_posix().replace("/", "_"))
            out_path = vis_dir / f"{image_stem}.jpg"

            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise RuntimeError(f"无法读取图片：{image_path}")

            panels: list[tuple[str, np.ndarray, str]] = []
            for model_path, model_stem, runner in runners:
                conf = resolve_vis_conf(
                    model_path,
                    detect_backend(model_path),
                    default=conf_default,
                    config_map=config_vis_conf,
                    cli_map=cli_vis_conf,
                )
                preds = runner.infer(image_path, conf, args.nms)
                panel = render_boxes_bgr(image_bgr, preds, [], runner.category_names)
                panels.append((model_stem, panel, rel_name))
                print(f"  {rel_name} | {model_stem}: {len(preds)} 个框 (conf≥{conf:g})")

            save_vis(panels, out_path, source_name=Path(rel_name).name)
            print(f"→ {out_path}")
    finally:
        for _, _, runner in runners:
            if isinstance(runner, HfRunner):
                runner.close()

    print(f"\n完成，共 {len(image_paths)} 张 → {vis_dir}/")


if __name__ == "__main__":
    main()
