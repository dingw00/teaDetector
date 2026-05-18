"""
数据增强概率默认值（等级 5 全强度基准）。

train_tea.py 在 configs/train.py 未定义 AUG_SIMPLE_* / AUG_DET_* 时，从此处读取。
等级 3/4 在训练时会对 detection 流水线概率与光度幅度再乘系数（见 utils/train.py LEVEL_DETECTION_STRENGTH）。
仅需在 configs/train.py 设置 AUG_LEVEL 即可；若要改单项概率，可在 train.py 中覆盖对应常量或 CLI 传参。
"""

# 等级 2（simple）
AUG_SIMPLE_FLIP_P = 0.5
AUG_SIMPLE_COLOR_P = 0.8

# 等级 4–5（detection 流水线，等级 5 为下列全值；3→×0.5，4→×0.75）
# 色彩：等级 5 用 RandomPhotometricDistort（亮度/对比度/饱和度/色相），由 PHOTOMETRIC_P 控制；
#       不走等级 2 的 ColorJitter（AUG_SIMPLE_COLOR_P 仅 aug_level=2）。
AUG_DET_PHOTOMETRIC_P = 0.9
AUG_DET_ZOOMOUT_FILL = 0.0
AUG_DET_ZOOMOUT_P = 0.9
AUG_DET_IOU_CROP_P = 0.9
AUG_DET_FLIP_P = 0.5
AUG_DET_MOSAIC_P = 0.7  # 仅 aug_level=5 且 train len>=4
