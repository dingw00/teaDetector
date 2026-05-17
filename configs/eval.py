"""茶叶目标检测评测配置（ONNX / HuggingFace checkpoint 共用）。"""

from pathlib import Path

from configs.preprocess import INPUT_SIZE

ROOT_DIR = Path(__file__).resolve().parent.parent

MODELS = [
    ROOT_DIR / "onnx_models" / "dino_0329_30.onnx",
    ROOT_DIR / "outputs" / "deimv2_s_march" / "checkpoint-best",
    ROOT_DIR / "outputs" / "deimv2_l_march" / "checkpoint-best",
    ROOT_DIR / "outputs" / "deimv2_l_march_and_april" / "checkpoint-best",
    ROOT_DIR / "outputs" / "deimv2_l_april" / "checkpoint-best",
]

DATASETS = [
    {"name": "teabud_march_ztu", "root": ROOT_DIR / "datasets" / "teabud_march_ztu"},
    {"name": "teabud_april", "root": ROOT_DIR / "datasets" / "teabud_april"},
    # {"name": "teabud_april_IY", "root": ROOT_DIR / "datasets" / "teabud_april_IY"},
]

NMS_THRESHOLD = 0.3
MAP_SCORE_THRESHOLD = 0.001
CONF_THRESHOLD = 0.2
MAP_IOU_THRESHOLDS = [x / 100 for x in range(50, 100, 5)]
VIS_NUM_IMAGES = 2
RANDOM_SEED = 42
OUTPUT_DIR = ROOT_DIR / "outputs" / "eval"
DEVICE = None
BATCH_SIZE = 2
NUM_WORKERS = 0
