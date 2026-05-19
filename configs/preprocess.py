"""

Deimv2 输入预处理（train_tea / eval_tea / export_onnx 共用；export 默认见 configs/export_onnx.py）。



几何缩放：

  RESIZE_MODE="stretch"  — HF processor 拉伸到 INPUT_SIZE×INPUT_SIZE（默认，与 hf_models.ipynb 一致）

  RESIZE_MODE="letterbox" — 等比缩放 + 黑边；训练在 dataloader 内 letterbox，processor.do_resize=False



像素：DO_RESCALE / DO_NORMALIZE / IMAGE_MEAN·STD。

命令行可用 utils.preprocess 中的 CLI 覆盖本文件默认值。

"""



from __future__ import annotations



from typing import Literal



INPUT_SIZE = 640



# "stretch" | "letterbox"

RESIZE_MODE: Literal["stretch", "letterbox"] = "letterbox"



# letterbox 画布填充色 (R, G, B)，与常见 OpenCV 黑边一致

LETTERBOX_FILL_RGB: tuple[int, int, int] = (0, 0, 0)



DO_RESIZE = True

DO_RESCALE = True

DO_NORMALIZE = True



IMAGE_MEAN = (0.485, 0.456, 0.406)

IMAGE_STD = (0.229, 0.224, 0.225)



USE_CHECKPOINT_PREPROCESSOR = True

FORCE_APPLY_CONFIG = False





def uses_letterbox() -> bool:

    return RESIZE_MODE == "letterbox"


