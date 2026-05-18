"""
Deimv2 输入预处理（train_tea / eval_tea / export_onnx 共用；export 默认见 configs/export_onnx.py）。

与官方 DEIMv2 hf_models.ipynb 对齐：Resize + ToTensor（÷255），默认不做 ImageNet mean/std。
命令行可用 utils.preprocess 中的 CLI 参数覆盖本文件默认值。
"""     

from __future__ import annotations

INPUT_SIZE = 640

DO_RESIZE = True
DO_RESCALE = True
DO_NORMALIZE = True

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

USE_CHECKPOINT_PREPROCESSOR = True
FORCE_APPLY_CONFIG = False
