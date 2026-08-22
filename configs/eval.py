"""茶叶目标检测评测配置（ONNX / HuggingFace checkpoint 共用）。"""

from pathlib import Path

from configs.preprocess import INPUT_SIZE

ROOT_DIR = Path(__file__).resolve().parent.parent

MODELS = [
    # ROOT_DIR / "onnx_models" / "dino_0329_30.onnx",
    # ROOT_DIR / "outputs" / "deimv2_s_march" / "checkpoint-best",
    ROOT_DIR / "outputs" / "2026pre_gy" / "checkpoint-best",
    # ROOT_DIR / "outputs" / "deimv2_l_april_then_marchapril" / "checkpoint-best",

]

DATASETS = [
    {"name": "2026pre_qm", "root": ROOT_DIR / "datasets" / "2026pre_qm"},
    {"name": "2026pre_gy", "root": ROOT_DIR / "datasets" / "2026pre_gy"},
    # {"name": "teabud_april", "root": ROOT_DIR / "datasets" / "teabud_april"},
]

# 仅用于 vis/ 抽样图绘制；HF/ONNX 的 mAP 不做额外 NMS，后处理 score 下限固定为 0。
NMS_THRESHOLD = 0.3
CONF_THRESHOLD = 0.2

# 可视化绘制 score 下限（按模型）；键为路径或 display 名中的子串，最长匹配优先。
# 未命中条目时使用 CONF_THRESHOLD 或命令行 --conf。
VIS_CONF_BY_MODEL: dict[str, float] = {
    "dino_0329_30": 0.10,
    "deimv2_l_march_checkpoint-best": 0.20,
    "deimv2_l_march_and_april_checkpoint-best": 0.20,
    "deimv2_l_april_checkpoint-best": 0.20,
}
MAP_IOU_THRESHOLDS = [x / 100 for x in range(50, 100, 5)]
# 仅 val 划分的可视化抽样张数（int 或 "all"）；train 固定 2 张。
# 命令行可用 --vis_num N 或 --vis_num all 覆盖。
VIS_NUM_IMAGES = "all"
# vis/ 多模型竖拼：每行保留原图分辨率，JPEG 质量 1–100。
# 评估时用不到，不指定。
VIS_JPEG_QUALITY = 95
RANDOM_SEED = 42
OUTPUT_DIR = ROOT_DIR / "outputs" / "eval"
DEVICE = None
BATCH_SIZE = 4
NUM_WORKERS = 0
