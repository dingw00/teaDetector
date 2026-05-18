"""茶叶目标检测评测配置（ONNX / HuggingFace checkpoint 共用）。"""

from pathlib import Path

from configs.preprocess import INPUT_SIZE

ROOT_DIR = Path(__file__).resolve().parent.parent

MODELS = [
    ROOT_DIR / "onnx_models" / "dino_0329_30.onnx",
    # ROOT_DIR / "outputs" / "deimv2_s_march" / "checkpoint-best",
    
    ROOT_DIR / "outputs" / "deimv2_l_march" / "checkpoint-best",
    # ROOT_DIR / "onnx_models" / "deimv2_l_march_checkpoint-best.onnx",
    # ROOT_DIR / "outputs" / "deimv2_l_march_and_april" / "checkpoint-best",
    ROOT_DIR / "outputs" / "deimv2_l_april_then_march" / "checkpoint-best",
    ROOT_DIR / "outputs" / "deimv2_l_april_then_marchapril" / "checkpoint-best",
    # ROOT_DIR / "onnx_models" / "deimv2_l_march_and_april_checkpoint-best.onnx",
    ROOT_DIR / "outputs" / "deimv2_l_april" / "checkpoint-best",
    # ROOT_DIR / "onnx_models" / "deimv2_l_april_checkpoint-best.onnx",

]

DATASETS = [
    {"name": "teabud_march_ztu", "root": ROOT_DIR / "datasets" / "teabud_march_ztu"},
    {"name": "teabud_april", "root": ROOT_DIR / "datasets" / "teabud_april"},
    # {"name": "teabud_april_IY", "root": ROOT_DIR / "datasets" / "teabud_april_IY"},
]

# 仅用于 vis/ 抽样图绘制；HF checkpoint 的 mAP 与 train_tea 一致，不做额外 NMS。
NMS_THRESHOLD = 0.3
# 与 configs/train.py MAP_SCORE_THRESHOLD 一致（HF post_process_object_detection）。
MAP_SCORE_THRESHOLD = 0.05
CONF_THRESHOLD = 0.2

# 可视化绘制 score 下限（按模型）；键为路径或 display 名中的子串，最长匹配优先。
# 未命中条目时使用 CONF_THRESHOLD 或命令行 --conf。
VIS_CONF_BY_MODEL: dict[str, float] = {
    "dino_0329_30": 0.10,
    "deimv2_s_march": 0.20,
    "deimv2_l_march": 0.20,
    "deimv2_l_march_checkpoint-best": 0.20,
    "deimv2_l_march_and_april": 0.20,
    "deimv2_l_march_and_april_checkpoint-best": 0.20,
    "deimv2_l_april": 0.20,
    "deimv2_l_april_checkpoint-best": 0.20,
}
MAP_IOU_THRESHOLDS = [x / 100 for x in range(50, 100, 5)]
VIS_NUM_IMAGES = 2
RANDOM_SEED = 42
OUTPUT_DIR = ROOT_DIR / "outputs" / "eval"
DEVICE = None
BATCH_SIZE = 2
NUM_WORKERS = 0
