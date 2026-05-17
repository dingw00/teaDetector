"""
train_tea.py 默认训练参数。

修改后运行 `python train_tea.py`；`--preset <name>` 套用 PRESETS。
命令行可覆盖任意项；预处理见 configs/preprocess.py；增强等级逻辑见 utils/augmentation.py。
"""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASETS: list[Path] = [
    ROOT_DIR / "datasets" / "teabud_march_ztu",
]
DATASET_RATIOS: list[float] | None = None
OUTPUT_DIR = ROOT_DIR / "outputs" / "deimv2_l"
RESUME_FROM: Path | None = None

EPOCHS = 100
BATCH_SIZE = 4
NUM_WORKERS = 0

PRETRAINED = "dinov3_l"
LR = 1e-4
WEIGHT_DECAY = 0.01
TRAIN_MODE = "backbone_frozen"
LOSS_BBOX_SCALE = 1.0
UNFREEZE_BACKBONE_LAST_N = 12
LR_BACKBONE = 1.25e-5
BACKBONE_LR_DECAY = 0.7
WARMUP_EPOCHS = 0
DEVICE: str | None = None

MAP_SCORE_THRESHOLD = 0.05
MAP_BATCH_SIZE: int | None = None

# 数据增强等级 1–5（默认 5 = 原 detection + Mosaic）
AUG_LEVEL = 5

# AUG_SIMPLE_FLIP_P = 0.5
# AUG_SIMPLE_COLOR_P = 0.8
# AUG_DET_PHOTOMETRIC_P = 0.7
# AUG_DET_ZOOMOUT_FILL = 0.0
# AUG_DET_ZOOMOUT_P = 0.6
# AUG_DET_IOU_CROP_P = 0.85
# AUG_DET_FLIP_P = 0.5
# AUG_DET_MOSAIC_P = 0.3  # 仅 aug_level=5：四宫格 Mosaic 概率

PRESETS: dict[str, dict] = {
    "dinov3_s_march": {
        "pretrained": "dinov3_s",
        "datasets": [ROOT_DIR / "datasets" / "teabud_march_ztu"],
        "output_dir": ROOT_DIR / "outputs" / "deimv2_s",
        "epochs": 200,
        "batch_size": 4,
        "unfreeze_backbone_last_n": 12,
        "lr_backbone": 1.25e-5,
        "backbone_lr_decay": 0.7,
    },
    "dinov3_l_march_april": {
        "pretrained": "dinov3_l",
        "datasets": [
            ROOT_DIR / "datasets" / "teabud_march_ztu",
            ROOT_DIR / "datasets" / "teabud_april",
        ],
        "output_dir": ROOT_DIR / "outputs" / "deimv2_l_march_and_april",
        "epochs": 200,
        "batch_size": 4,
        "unfreeze_backbone_last_n": 12,
        "lr_backbone": 1.25e-5,
        "backbone_lr_decay": 0.7,
    },
}
