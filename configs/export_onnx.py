"""
export_onnx.py 默认参数。

修改后运行 `python export_onnx.py`（若已设置 CHECKPOINT）；
命令行可覆盖任意项。输入尺寸与预处理见 configs/preprocess.py。
"""

from __future__ import annotations

from pathlib import Path

from configs.preprocess import INPUT_SIZE

ROOT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT: Path | None = None # outputs/deimv2_s_march/checkpoint-best

OUTPUT: Path | None = None  # 未设则 onnx_models/<run>_epochN.onnx（不用 checkpoint-best 作文件名）
OPSET = 17
INPUT_SIZE = INPUT_SIZE
DYNAMIC_BATCH = False
VERIFY = True
CHECK = True
SIMPLIFY = False
