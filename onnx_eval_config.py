"""ONNX 目标检测评测配置。"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

# 模型和数据集
ONNX_MODEL_PATH = ROOT_DIR / "onnx" / "dino_0329_30.onnx"
DATASET_ROOT = ROOT_DIR / "datasets" / "teabud_dataset_ztu"
IMAGE_DIR = DATASET_ROOT / "images"
TRAIN_ANN = DATASET_ROOT / "annotations" / "train_30.json"
VAL_ANN = DATASET_ROOT / "annotations" / "val_30_30.json"

# 与 C++ 推理代码一致：输入固定为 input_size x input_size
INPUT_SIZE = 640

# 后处理阈值
CONF_THRESHOLD = 0.1
NMS_THRESHOLD = 0.3

# 指标和可视化
MAP_IOU_THRESHOLDS = [x / 100 for x in range(50, 100, 5)]
VIS_NUM_IMAGES = 2
RANDOM_SEED = 42
OUTPUT_DIR = ROOT_DIR / "outputs" / "onnx_eval"
