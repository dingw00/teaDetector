"""茶叶目标检测评测配置（ONNX / HuggingFace checkpoint 共用）。"""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

# 默认评测模型（eval_tea.py 未指定 --model 时使用）
MODEL_PATH = ROOT_DIR / "onnx_models" / "dino_0329_30.onnx"
DEFAULT_MODELS = [MODEL_PATH]
# DEFAULT_MODELS = [
#     MODEL_PATH,
#     ROOT_DIR / "outputs" / "deimv2_s_tealeaves" / "final",
# ]

# 多数据集：每个 root 下需有 images/ 与 annotations/train.json、val.json
DATASETS = [
    {
        "name": "TeaLeavesDatasets_split_lr",
        "root": ROOT_DIR / "datasets" / "TeaLeavesDatasets_split_lr",
    },
    # {
    #     "name": "teabud_dataset_ztu",
    #     "root": ROOT_DIR / "datasets" / "teabud_dataset_ztu",
    # },
]

# ONNX：与 C++ 推理一致
INPUT_SIZE = 640

NMS_THRESHOLD = 0.3
CONF_THRESHOLD = 0.2
HF_CONF_THRESHOLD = 0.05

MAP_IOU_THRESHOLDS = [x / 100 for x in range(50, 100, 5)]
VIS_NUM_IMAGES = 2
RANDOM_SEED = 42
OUTPUT_DIR = ROOT_DIR / "outputs" / "eval"

HF_BATCH_SIZE = 2
HF_NUM_WORKERS = 0
HF_DEVICE = None
