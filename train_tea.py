"""
使用 DEIMv2（DINOv3 骨干）在茶叶 COCO 数据集上做迁移学习。

通过 --pretrained 从 HuggingFace Hub 拉取 Transformers 版 Deimv2 权重（默认 DINOv3-S；
可选 DINOv3-L 等大骨干）。Intellindust/DEIMv2_DINOv3_L_COCO 等为原版仓库配置，不能用于本脚本的 from_pretrained。

使用 HuggingFace Transformers 中的 Deimv2ForObjectDetection。
输入预处理见 configs/preprocess.py（train_tea / eval_tea / export_onnx 共用）。

默认（--train_mode backbone_frozen）：只冻结 conv_encoder 内的 backbone（DINOv3 为
model.conv_encoder.backbone.*，CNN 骨干为 model.conv_encoder.model.*）；neck（如 STA、
fusion_proj）、HybridEncoder、Decoder、分类与框回归等其余参数全部训练，并使用完整
检测损失。可用 --loss_bbox_scale 放大 bbox/giou 权重。
可选 --unfreeze_backbone_last_n：在各类 train_mode 的冻结策略之后，对 DINOv3 ViT 骨干再解冻最后若干 Transformer block；
当 n>0 且骨干为 dinov3_vit 时，自动对「可训练的」conv_encoder.backbone.* 使用按 block 分层学习率（lr_backbone、backbone_lr_decay），
与 deimv2_dinov3_l_coco.yml 思路一致：该部分 norm/bias 使用 weight_decay=0，其余权重使用 --weight_decay；检测头/neck 等仍用 --lr。
可选 --warmup_epochs：前若干个 epoch 对骨干参数组（conv_encoder.backbone / conv_encoder.model）线性 lr 热身（非骨干组不使用）。

其它模式：--train_mode heads_only 仅训练各检测头（仍冻 encoder/decoder 主体）；
--train_mode classification_only 为仅分类头且 bbox 类损失置 0（易 mAP 接近 0）。

断点续训：使用 --resume_from 指向某次保存目录（如 checkpoint-epoch5）。若该目录含
training_state.pt（每轮保存时会写入），将恢复优化器与 RNG；否则仅加载模型权重，
并根据 train_metrics.json 或目录名推断下一 epoch。续训时不会读取 output_dir 下已有
checkpoint-best/ 的历史最佳，仅以 resume 目录内 train_metrics.json 的 val mAP 为更新
checkpoint-best 的基准（若无则从零开始比较）。

每轮结束会写入 checkpoint-epochN，并同步到 final/（与最近一轮权重一致，便于中断后续训仍可用）。
验证集 bbox mAP 创新高时另存 checkpoint-best/。

依赖：torch, torchvision, transformers, scipy（Hungarian 匹配需要）, pycocotools（验证 mAP）

训练数据增强由 --aug_level 选择（1–5，默认见 configs/train.py），等级说明见 utils/train.py 开头。
验证/mAP 始终无增强。

多数据集：--datasets 可传一个或多个 COCO 根目录。单个时直接训练；多个时合并 train 用于训练，
各目录 val 在每轮单独算 mAP 并打印，最后输出各指标平均值。
检测类别数：合并各集 train 标注中的全部类别（按 category_id 去重），集合大小即 num_labels；
训练时将 COCO category_id 映射为 0..num_labels-1，mAP 时再映回 COCO id。
"""

from __future__ import annotations

import argparse
import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import torch
from pycocotools.coco import COCO
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Deimv2ForObjectDetection

from transformers.utils import logging as hf_logging

import configs.augmentation as _aug
import configs.train as _tc
from utils.common import HF_DEIMV2_PRESETS, resolve_pretrained_hub_id
from utils.preprocess import (
    add_preprocess_arguments,
    apply_preprocess_from_namespace,
    describe_processor,
    load_deimv2_processor,
)
from utils.train import (
    _infer_completed_epoch,
    _load_best_val_bbox_map,
    _load_training_state_path,
    _load_val_bbox_map_from_metrics,
    _promote_checkpoint_dir,
    _save_epoch_checkpoint,
    _sum_weighted_cls_loss,
    _sum_weighted_giou_loss,
    _sum_weighted_l1_bbox_loss,
    _val_bbox_map,
    apply_adamw_named_state,
    apply_backbone_linear_warmup_lrs,
    apply_dinov3_backbone_last_n_unfreeze,
    apply_train_mode,
    augmentation_metrics_block,
    build_adamw_param_groups,
    build_train_val_sources,
    config_num_labels,
    evaluate_coco_bbox_map,
    evaluate_val_maps_per_dataset,
    resolve_categories_from_dataset_roots,
    format_augmentation_log_line,
    make_train_collate_fn,
    move_labels_to_device,
    parse_aug_level,
    resolve_cli_path,
    resolve_train_mosaic_p,
    sync_optimizer_param_group_metadata,
    _dinov3_vit_num_hidden_layers,
)

hf_logging.disable_progress_bar()
_SCRIPT_DIR = Path(__file__).resolve().parent

# configs/train.py 顶层属性 → argparse 目标名（PRESETS 内键名与右侧一致）
_TC_ARG_FIELDS: tuple[tuple[str, str], ...] = (
    ("datasets", "DATASETS"),
    ("dataset_ratios", "DATASET_RATIOS"),
    ("output_dir", "OUTPUT_DIR"),
    ("resume_from", "RESUME_FROM"),
    ("epochs", "EPOCHS"),
    ("batch_size", "BATCH_SIZE"),
    ("num_workers", "NUM_WORKERS"),
    ("pretrained", "PRETRAINED"),
    ("lr", "LR"),
    ("train_mode", "TRAIN_MODE"),
    ("loss_bbox_scale", "LOSS_BBOX_SCALE"),
    ("unfreeze_backbone_last_n", "UNFREEZE_BACKBONE_LAST_N"),
    ("lr_backbone", "LR_BACKBONE"),
    ("backbone_lr_decay", "BACKBONE_LR_DECAY"),
    ("weight_decay", "WEIGHT_DECAY"),
    ("warmup_epochs", "WARMUP_EPOCHS"),
    ("device", "DEVICE"),
    ("map_score_threshold", "MAP_SCORE_THRESHOLD"),
    ("map_batch_size", "MAP_BATCH_SIZE"),
    ("aug_level", "AUG_LEVEL"),
    ("aug_simple_flip_p", "AUG_SIMPLE_FLIP_P"),
    ("aug_simple_color_p", "AUG_SIMPLE_COLOR_P"),
    ("aug_det_photometric_p", "AUG_DET_PHOTOMETRIC_P"),
    ("aug_det_zoomout_fill", "AUG_DET_ZOOMOUT_FILL"),
    ("aug_det_zoomout_p", "AUG_DET_ZOOMOUT_P"),
    ("aug_det_iou_crop_p", "AUG_DET_IOU_CROP_P"),
    ("aug_det_flip_p", "AUG_DET_FLIP_P"),
    ("aug_det_mosaic_p", "AUG_DET_MOSAIC_P"),
)
_PRESET_ARG_NAMES = frozenset(name for name, _ in _TC_ARG_FIELDS)

# configs/train.py 可省略；未定义时从 configs/augmentation.py 读取（与 AUG_LEVEL 配套）
_AUG_TC_ATTRS = frozenset(
    tc_attr
    for arg_name, tc_attr in _TC_ARG_FIELDS
    if arg_name.startswith("aug_") and arg_name != "aug_level"
)


def _config_value(tc_attr: str) -> object:
    if hasattr(_tc, tc_attr):
        return getattr(_tc, tc_attr)
    if tc_attr in _AUG_TC_ATTRS and hasattr(_aug, tc_attr):
        return getattr(_aug, tc_attr)
    raise AttributeError(
        f"configs.train 与 configs.augmentation 均未定义 {tc_attr!r}；"
        f"请设置 AUG_LEVEL 或于 configs/augmentation.py 中补充默认值。"
    )


def _training_defaults(preset: str | None) -> dict[str, object]:
    """合并 configs/train.py 顶层默认与 PRESETS[preset]（preset 中的键覆盖顶层）。"""
    cfg: dict[str, object] = {}
    for arg_name, tc_attr in _TC_ARG_FIELDS:
        val = _config_value(tc_attr)
        cfg[arg_name] = list(val) if arg_name == "datasets" else val
    if preset is None:
        return cfg
    if preset not in _tc.PRESETS:
        raise ValueError(f"未知 --preset={preset!r}，可选: {tuple(_tc.PRESETS)}")
    overlay = _tc.PRESETS[preset]
    unknown = set(overlay) - _PRESET_ARG_NAMES
    if unknown:
        raise ValueError(f"PRESETS[{preset!r}] 含未识别键: {sorted(unknown)}")
    for key, val in overlay.items():
        cfg[key] = list(val) if key == "datasets" else val
    return cfg


def parse_args(argv: list[str] | None = None):
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument(
        "--preset",
        choices=tuple(_tc.PRESETS),
        default=None,
        help=argparse.SUPPRESS,
    )
    preset_args, argv_rest = preset_parser.parse_known_args(argv)

    d = _training_defaults(preset_args.preset)
    device_default = d["device"] if d["device"] is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    p = argparse.ArgumentParser(
        description="茶叶 DEIMv2 训练；可用 --preset 套用 configs/train.py PRESETS，其它 CLI 参数覆盖 preset。",
    )
    p.add_argument(
        "--preset",
        choices=tuple(_tc.PRESETS),
        default=preset_args.preset,
        help=f"套用 configs/train.py PRESETS（可选: {', '.join(_tc.PRESETS)}）；未指定的项仍用顶层默认",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        type=Path,
        default=d["datasets"],
        metavar="DIR",
        help="一个或多个 COCO 数据集根目录（含 images/ 与 annotations/）。"
        "传 1 个则单集训练；传多个则合并 train 训练，每轮对各集 val 分别评测并打印均值。"
        "相对路径相对于本脚本所在目录。",
    )
    p.add_argument(
        "--dataset_ratios",
        nargs="*",
        type=float,
        default=d["dataset_ratios"],
        metavar="RATIO",
        help="与 --datasets 一一对应的训练采样比例（仅多个数据集时生效）。"
        "例如 1 2 表示两集被抽到的概率比为 1:2，与各自图片张数无关。"
        "不写则按各集 train 图片数量自然配比（与合并后均匀 shuffle 一致）。",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=d["output_dir"],
        help="checkpoint 与日志输出目录。相对路径相对于本脚本所在目录",
    )
    p.add_argument("--epochs", type=int, default=d["epochs"])
    p.add_argument("--batch_size", type=int, default=d["batch_size"])
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument(
        "--pretrained",
        type=str,
        choices=tuple(HF_DEIMV2_PRESETS),
        default=d["pretrained"],
        help="HF Hub 上的 DEIMv2+DINOv3 预训练预设。dinov3_l 对应社区 Transformers 版大骨干权重。",
    )
    p.add_argument("--num_workers", type=int, default=d["num_workers"])
    p.add_argument(
        "--train_mode",
        type=str,
        choices=("backbone_frozen", "heads_only", "classification_only"),
        default=d["train_mode"],
        help="backbone_frozen=只冻骨干其余全训；heads_only=只训检测头；classification_only=仅分类且框损失为 0",
    )
    p.add_argument(
        "--loss_bbox_scale",
        type=float,
        default=d["loss_bbox_scale"],
        help="在 backbone_frozen / heads_only 下，将 bbox 与 giou 损失权重相对默认再乘该系数",
    )
    p.add_argument(
        "--unfreeze_backbone_last_n",
        type=int,
        default=d["unfreeze_backbone_last_n"],
        help="仅骨干为 DINOv3 ViT 时生效：在 train_mode 冻结策略之后，再解冻 backbone 最后 n 个 Transformer block（及 conv_encoder.backbone.norm）。"
        "n>0 时自动仅对可训练的 conv_encoder.backbone.* 使用 lr_backbone / backbone_lr_decay 分层与 norm/bias 的 weight_decay=0；"
        "检测头/neck 等仍用 --lr。0=关闭。可大于总层数，将按 num_hidden_layers 封顶。",
    )
    p.add_argument(
        "--lr_backbone",
        type=float,
        default=d["lr_backbone"],
        help="与 --unfreeze_backbone_last_n>0 且 DINOv3 骨干联用：最后一层 ViT block 与 conv_encoder.backbone.norm 的学习率；默认 0.125×--lr",
    )
    p.add_argument(
        "--backbone_lr_decay",
        type=float,
        default=d["backbone_lr_decay"],
        help="与分层骨干联用：由深到浅每浅一层多乘该因子；1.0 表示各解冻 block 均为 lr_backbone",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=d["weight_decay"],
        help="AdamW 的 weight_decay；在启用骨干分层时，对 conv_encoder.backbone 中 norm/bias 等自动置 0，其余权重与检测部分仍用该值",
    )
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=d["warmup_epochs"],
        help="前若干个 epoch 仅对骨干参数组（conv_encoder.backbone / conv_encoder.model）做线性 lr 热身："
        "mult=(epoch-1)/(W-1) 自 0→1，lr=base_lr×mult；非骨干组恒为 base_lr。0=关闭。",
    )
    p.add_argument(
        "--device",
        type=str,
        default=device_default,
    )
    p.add_argument(
        "--map_score_threshold",
        type=float,
        default=d["map_score_threshold"],
        help="验证 mAP 时过滤低分框的阈值（与训练无关）",
    )
    p.add_argument(
        "--map_batch_size",
        type=int,
        default=d["map_batch_size"],
        help="算 mAP 时的 batch，默认与 --batch_size 相同",
    )
    p.add_argument(
        "--resume_from",
        type=Path,
        default=d["resume_from"],
        help="从某次保存的目录恢复（含 config.json、模型权重）。若同目录有 training_state.pt 则一并恢复优化器并从下一 epoch 继续；不加载 output_dir/checkpoint-best 的历史最佳；相对路径相对于本脚本所在目录",
    )
    p.add_argument(
        "--aug_level",
        type=parse_aug_level,
        default=parse_aug_level(d["aug_level"]),
        metavar="N",
        help="数据增强等级 1–5：1无 / 2simple(翻转+颜色) / 3–4检测v2弱档 / 5检测v2全强度+可选Mosaic（默认）。"
        "亦接受旧名 none|simple|normal|detection|detector。详见 utils/augmentation.py",
    )
    p.add_argument(
        "--aug_simple_flip_p",
        type=float,
        default=d["aug_simple_flip_p"],
        help="等级 2–3：水平翻转概率",
    )
    p.add_argument(
        "--aug_simple_color_p",
        type=float,
        default=d["aug_simple_color_p"],
        help="等级 3：ColorJitter 应用概率",
    )
    p.add_argument(
        "--aug_det_photometric_p",
        type=float,
        default=d["aug_det_photometric_p"],
        help="等级 4–5：RandomPhotometricDistort 概率",
    )
    p.add_argument(
        "--aug_det_zoomout_fill",
        type=float,
        default=d["aug_det_zoomout_fill"],
        help="等级 4–5：RandomZoomOut 的 fill",
    )
    p.add_argument(
        "--aug_det_zoomout_p",
        type=float,
        default=d["aug_det_zoomout_p"],
        help="等级 4–5：RandomZoomOut 概率",
    )
    p.add_argument(
        "--aug_det_iou_crop_p",
        type=float,
        default=d["aug_det_iou_crop_p"],
        help="等级 4–5：RandomIoUCrop 以 RandomApply 包裹的应用概率",
    )
    p.add_argument(
        "--aug_det_flip_p",
        type=float,
        default=d["aug_det_flip_p"],
        help="等级 4–5：RandomHorizontalFlip 概率",
    )
    p.add_argument(
        "--aug_det_mosaic_p",
        type=float,
        default=d["aug_det_mosaic_p"],
        help="仅 aug_level=5 且训练集 len>=4：四宫格 Mosaic 概率（默认见 configs/augmentation.py）",
    )
    add_preprocess_arguments(p)
    return p.parse_args(argv_rest)

def main():
    args = parse_args()
    prep_applied = apply_preprocess_from_namespace(args)
    if prep_applied:
        print("预处理 CLI 覆盖: " + ", ".join(prep_applied))
    if args.preset:
        print(f"已套用配置预设: {args.preset}（定义见 configs/train.py PRESETS）")
    args.datasets = [resolve_cli_path(p) for p in args.datasets]
    if not args.datasets:
        raise ValueError("--datasets 至少指定一个 COCO 数据集根目录")
    args.output_dir = resolve_cli_path(args.output_dir)
    if args.resume_from is not None:
        args.resume_from = resolve_cli_path(args.resume_from)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.warmup_epochs < 0:
        raise ValueError("--warmup_epochs 不能为负数")

    category_spec = resolve_categories_from_dataset_roots(args.datasets)
    print(
        f"检测类别（合并 {len(args.datasets)} 个训练集）: num_labels={category_spec.num_labels}（"
        f"共 {len(category_spec.coco_id_to_label)} 个 COCO category_id）"
    )
    for coco_id in sorted(category_spec.coco_id_to_label):
        lab = category_spec.coco_id_to_label[coco_id]
        print(f"  label {lab} ← COCO id={coco_id} {category_spec.id2label[lab]!r}")

    start_epoch = 0
    resume_training_state: dict | None = None
    if args.resume_from is not None:
        resume_dir = args.resume_from
        if not resume_dir.is_dir():
            raise FileNotFoundError(f"--resume_from 不是目录: {resume_dir}")
        processor = load_deimv2_processor(resume_dir)
        model = Deimv2ForObjectDetection.from_pretrained(str(resume_dir))
        ckpt_labels = config_num_labels(model.config)
        if ckpt_labels != category_spec.num_labels:
            raise ValueError(
                f"续训 checkpoint 类别数 num_labels={ckpt_labels} 与当前 --datasets 解析结果 "
                f"{category_spec.num_labels} 不一致；请使用与训练时相同的数据集组合。"
            )
        ts_path = _load_training_state_path(resume_dir)
        if ts_path is not None:
            resume_training_state = torch.load(ts_path, map_location="cpu", weights_only=False)
            start_epoch = int(resume_training_state["epoch"])
        else:
            inferred = _infer_completed_epoch(resume_dir)
            start_epoch = int(inferred) if inferred is not None else 0
    else:
        pretrained_hub = resolve_pretrained_hub_id(args.pretrained)
        print(f"从 HuggingFace Hub 加载预训练: {pretrained_hub}")
        processor = load_deimv2_processor(pretrained_hub)
        print(f"图像预处理: {describe_processor(processor)}  （配置见 configs/preprocess.py）")
        model = Deimv2ForObjectDetection.from_pretrained(
            pretrained_hub,
            num_labels=category_spec.num_labels,
            ignore_mismatched_sizes=True,
        )
        model.config.id2label = dict(category_spec.id2label)
        model.config.label2id = dict(category_spec.label2id)
        model.config.num_labels = category_spec.num_labels

    load_source = str(args.resume_from) if args.resume_from is not None else resolve_pretrained_hub_id(args.pretrained)

    apply_train_mode(model, args.train_mode, args.loss_bbox_scale)
    apply_dinov3_backbone_last_n_unfreeze(model, args.unfreeze_backbone_last_n)
    print(f"train_mode={args.train_mode}，可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    train_mosaic_p = resolve_train_mosaic_p(args.aug_level, mosaic_p=args.aug_det_mosaic_p)
    aug_record = augmentation_metrics_block(
        args.aug_level,
        simple_flip_p=args.aug_simple_flip_p,
        simple_color_p=args.aug_simple_color_p,
        det_photometric_p=args.aug_det_photometric_p,
        det_zoomout_fill=args.aug_det_zoomout_fill,
        det_zoomout_p=args.aug_det_zoomout_p,
        det_iou_crop_p=args.aug_det_iou_crop_p,
        det_flip_p=args.aug_det_flip_p,
        mosaic_p=args.aug_det_mosaic_p,
    )
    print("训练数据增强 " + format_augmentation_log_line(aug_record))

    train_ds, train_ann, dataset_roots, merged_coco_cache, val_eval_sources, train_sample_weights = (
        build_train_val_sources(
            args,
            script_dir=_SCRIPT_DIR,
            category_id_remap=category_spec.coco_id_to_label,
        )
    )
    print(f"训练数据: {'合并 train' if len(args.datasets) > 1 else '单集'}，共 {len(args.datasets)} 个数据集根目录:")
    for i, r in enumerate(dataset_roots):
        print(f"  [{i}] {r}")
    if merged_coco_cache is not None:
        print(f"合并 COCO 缓存目录: {merged_coco_cache}")
    print("每轮 val mAP 将按以下子集分别评测" + ("并求平均" if len(val_eval_sources) > 1 else "") + ":")
    for src in val_eval_sources:
        print(f"  - {src.name} ({src.ann_path})")

    if train_mosaic_p > 0 and len(train_ds) < 4:
        print("警告: aug_level=5 且 mosaic_p>0 但训练集长度 <4，Mosaic 分支不会触发。")

    device = torch.device(args.device)
    model.to(device)

    buf = StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        coco_gt_train = COCO(str(train_ann))
    map_bs = args.map_batch_size if args.map_batch_size is not None else args.batch_size

    train_collate = make_train_collate_fn(
        processor,
        aug_level=args.aug_level,
        simple_flip_p=args.aug_simple_flip_p,
        simple_color_p=args.aug_simple_color_p,
        det_photometric_p=args.aug_det_photometric_p,
        det_zoomout_fill=args.aug_det_zoomout_fill,
        det_zoomout_p=args.aug_det_zoomout_p,
        det_iou_crop_p=args.aug_det_iou_crop_p,
        det_flip_p=args.aug_det_flip_p,
    )

    if train_sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(train_sample_weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=train_collate,
            pin_memory=device.type == "cuda",
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=train_collate,
            pin_memory=device.type == "cuda",
        )

    if args.backbone_lr_decay <= 0:
        raise ValueError("--backbone_lr_decay 必须 > 0")
    lr_backbone_eff = args.lr_backbone if args.lr_backbone is not None else args.lr * 0.125
    param_groups, used_layerwise = build_adamw_param_groups(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        unfreeze_backbone_last_n=args.unfreeze_backbone_last_n,
        lr_backbone=lr_backbone_eff,
        backbone_lr_decay=args.backbone_lr_decay,
    )
    opt = AdamW(param_groups)
    if used_layerwise:
        nl = _dinov3_vit_num_hidden_layers(model)
        print(
            f"优化器: 因 unfreeze_backbone_last_n={args.unfreeze_backbone_last_n}>0，对 DINOv3 可训练 backbone 分层 lr"
            f"（ViT block 数={nl}），lr_backbone={lr_backbone_eff:g}，由深到浅每层×{args.backbone_lr_decay:g}；"
            f"检测头/neck 等 lr={args.lr:g}；weight_decay={args.weight_decay:g}（backbone 内 norm/bias 等 wd=0），"
            f"param_groups={len(param_groups)}"
        )
        backbone_trainable = any(
            (".conv_encoder.backbone." in n or ".conv_encoder.model." in n)
            for n, p in model.named_parameters()
            if p.requires_grad
        )
        if not backbone_trainable:
            print(
                "警告: 已请求骨干分层 lr，但当前无可训练 conv_encoder 骨干参数（请确认 unfreeze 是否生效）。"
            )
    else:
        print(f"优化器: AdamW lr={args.lr:g}, weight_decay={args.weight_decay:g}, param_groups={len(param_groups)}")

    opt_resume = "none"  # full | named | fresh | none
    if resume_training_state is not None:
        if "optimizer" in resume_training_state:
            try:
                opt.load_state_dict(resume_training_state["optimizer"])
                opt_resume = "full"
            except Exception as exc:
                print(f"警告: 未能整表加载优化器 state_dict（param_groups 可能已变）: {exc}")
                n_named = apply_adamw_named_state(
                    opt, model, resume_training_state.get("optimizer_named_state")
                )
                opt_resume = "named" if n_named > 0 else "fresh"
                if n_named > 0:
                    print(
                        f"提示: 已按参数名合并 AdamW 动量到 {n_named} 个参数；"
                        "新解冻或新分组中的参数无历史动量。"
                    )
                else:
                    print("警告: checkpoint 中无可用 optimizer_named_state（旧格式），优化器动量已重新初始化。")
                    print(
                        "提示: 自本脚本起保存的 training_state.pt 会附带 optimizer_named_state；"
                        "若需今后再改 param_groups/解冻层数并续接动量，请从含该字段的 checkpoint 续训。"
                    )
        if resume_training_state.get("rng_cpu") is not None:
            torch.random.set_rng_state(resume_training_state["rng_cpu"])
        if resume_training_state.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume_training_state["rng_cuda"])

    if args.resume_from is not None:
        if resume_training_state is not None:
            if opt_resume == "full":
                what = "模型、优化器与 RNG"
            elif opt_resume == "named":
                what = "模型、RNG，以及按参数名合并的 AdamW 动量（param_groups 与保存时不一致）"
            else:
                what = "模型与 RNG（优化器为当前超参下的新 AdamW）"
            print(
                f"已从 {args.resume_from} 恢复{what}，"
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

    sync_optimizer_param_group_metadata(
        opt,
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        unfreeze_backbone_last_n=args.unfreeze_backbone_last_n,
        lr_backbone=lr_backbone_eff,
        backbone_lr_decay=args.backbone_lr_decay,
    )
    if args.warmup_epochs > 0:
        if any(g.get("is_backbone") for g in opt.param_groups):
            print(
                f"骨干线性热身: {args.warmup_epochs} 个 epoch（epoch 1 时 mult=0，epoch {args.warmup_epochs} 时 mult=1），"
                "仅 conv_encoder 骨干参数组；检测头等保持 base_lr。"
            )
        else:
            print("提示: --warmup_epochs>0 但当前无可训练骨干参数，热身未生效。")

    best_val_bbox_map: float | None = None
    best_val_epoch: int | None = None
    if args.resume_from is not None:
        best_val_bbox_map, best_val_epoch = _load_val_bbox_map_from_metrics(
            args.resume_from / "train_metrics.json"
        )
        if best_val_bbox_map is not None:
            print(
                f"续训基准（来自 {args.resume_from.name}）: epoch {best_val_epoch}, "
                f"val bbox mAP={best_val_bbox_map:.4f}；不读取 checkpoint-best 历史最佳"
            )
    else:
        best_val_bbox_map, best_val_epoch = _load_best_val_bbox_map(args.output_dir)
        if best_val_bbox_map is not None:
            print(
                f"已加载历史最佳 checkpoint-best: epoch {best_val_epoch}, "
                f"val bbox mAP={best_val_bbox_map:.4f}"
            )

    for epoch in range(next_epoch, args.epochs + 1):
        m_warm = apply_backbone_linear_warmup_lrs(opt, epoch, args.warmup_epochs)
        model.train()
        running = 0.0
        running_cls = 0.0
        running_l1_bbox = 0.0
        running_giou = 0.0
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
            ld = getattr(out, "loss_dict", None)
            running_cls += _sum_weighted_cls_loss(ld)
            running_l1_bbox += _sum_weighted_l1_bbox_loss(ld)
            running_giou += _sum_weighted_giou_loss(ld)
            n_batches += 1

        train_loss = running / max(n_batches, 1)
        train_loss_cls = running_cls / max(n_batches, 1)
        train_loss_l1_bbox = running_l1_bbox / max(n_batches, 1)
        train_loss_giou = running_giou / max(n_batches, 1)

        map_train = evaluate_coco_bbox_map(
            model,
            processor,
            train_ds,
            coco_gt_train,
            device,
            batch_size=map_bs,
            score_threshold=args.map_score_threshold,
            num_workers=args.num_workers,
            label_to_coco_id=category_spec.label_to_coco_id,
        )
        map_val_per, map_val_mean = evaluate_val_maps_per_dataset(
            model,
            processor,
            val_eval_sources,
            device,
            batch_size=map_bs,
            score_threshold=args.map_score_threshold,
            num_workers=args.num_workers,
            label_to_coco_id=category_spec.label_to_coco_id,
        )
        map_val = map_val_mean if len(val_eval_sources) > 1 else next(iter(map_val_per.values()))

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} | cls≈{train_loss_cls:.4f} | "
            f"bbox(L1)≈{train_loss_l1_bbox:.4f} | giou≈{train_loss_giou:.4f}"
            + (f" | bb_warmup={m_warm:.3f}" if args.warmup_epochs > 0 else "")
        )
        print(
            f"  train mAP={map_train['bbox_mAP']:.4f} mAP50={map_train['bbox_mAP_50']:.4f} "
            f"mAP75={map_train['bbox_mAP_75']:.4f} AR100={map_train['bbox_mAR_100']:.4f}"
        )

        save_dir = args.output_dir / f"checkpoint-epoch{epoch}"
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_cls_weighted": train_loss_cls,
            "train_loss_bbox_l1_weighted": train_loss_l1_bbox,
            "train_loss_giou_weighted": train_loss_giou,
            "map_score_threshold": args.map_score_threshold,
            "train_map": map_train,
            "val_map": map_val,
            "val_map_per_dataset": map_val_per,
            "val_map_mean": map_val_mean if len(val_eval_sources) > 1 else None,
            "train_mode": args.train_mode,
            "unfreeze_backbone_last_n": args.unfreeze_backbone_last_n,
            "warmup_epochs": args.warmup_epochs,
            "backbone_warmup_mult": m_warm,
            "pretrained": args.pretrained,
            "load_source": load_source,
            "dataset_roots": dataset_roots,
            "num_labels": category_spec.num_labels,
            "id2label": {str(k): v for k, v in category_spec.id2label.items()},
            "coco_id_to_label": {str(k): v for k, v in category_spec.coco_id_to_label.items()},
            "dataset_ratios": list(args.dataset_ratios) if args.dataset_ratios else None,
            "train_sample_weights_enabled": train_sample_weights is not None,
            "merged_coco_cache": str(merged_coco_cache) if merged_coco_cache is not None else None,
            "optimizer": {
                "layerwise_dinov3_backbone_lr": used_layerwise,
                "lr": args.lr,
                "lr_backbone": lr_backbone_eff if used_layerwise else None,
                "backbone_lr_decay": args.backbone_lr_decay if used_layerwise else None,
                "weight_decay": args.weight_decay,
                "param_groups": len(param_groups),
            },
            "augmentation": augmentation_metrics_block(
                args.aug_level,
                simple_flip_p=args.aug_simple_flip_p,
                simple_color_p=args.aug_simple_color_p,
                det_photometric_p=args.aug_det_photometric_p,
                det_zoomout_fill=args.aug_det_zoomout_fill,
                det_zoomout_p=args.aug_det_zoomout_p,
                det_iou_crop_p=args.aug_det_iou_crop_p,
                det_flip_p=args.aug_det_flip_p,
                mosaic_p=args.aug_det_mosaic_p,
            ),
            "loss_weights": {
                "mal": model.config.weight_loss_mal,
                "bbox": model.config.weight_loss_bbox,
                "giou": model.config.weight_loss_giou,
                "fgl": model.config.weight_loss_fgl,
                "ddf": model.config.weight_loss_ddf,
            },
        }
        _save_epoch_checkpoint(
            save_dir,
            model=model,
            processor=processor,
            metrics=metrics,
            opt=opt,
            epoch=epoch,
            unfreeze_backbone_last_n=args.unfreeze_backbone_last_n,
            warmup_epochs=args.warmup_epochs,
            pretrained=args.pretrained,
            load_source=load_source,
        )
        _promote_checkpoint_dir(save_dir, args.output_dir / "final")

        val_bbox_map = _val_bbox_map(map_val if isinstance(map_val, dict) else None)
        if val_bbox_map is not None and (
            best_val_bbox_map is None or val_bbox_map >= best_val_bbox_map
        ):
            best_val_bbox_map = val_bbox_map
            best_val_epoch = epoch
            best_dir = args.output_dir / "checkpoint-best"
            _promote_checkpoint_dir(save_dir, best_dir)
            print(
                f"  新最佳 val bbox mAP={val_bbox_map:.4f}，已保存到 {best_dir.name}/"
            )

    final_dir = args.output_dir / "final"
    best_dir = args.output_dir / "checkpoint-best"
    print(f"训练结束：最近一轮权重在 {final_dir}（与 checkpoint-epoch{epoch} 一致）")
    if best_val_epoch is not None:
        print(
            f"验证集最佳: epoch {best_val_epoch}, val bbox mAP={best_val_bbox_map:.4f} -> {best_dir}"
        )
    else:
        print("未记录 checkpoint-best（本轮无有效 val bbox mAP）")


if __name__ == "__main__":
    main()
