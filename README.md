.venv\Scripts\Activate.ps1

python train_tea.py --pretrained dinov3_l --datasets ./datasets/teabud_dataset_ztu --output_dir ./outputs/deimv2_l --batch_size 4 --lr 0.0001 --train_mode backbone_frozen --unfreeze_backbone_last_n 12 --lr_backbone 0.0000125 --backbone_lr_decay 0.7 --resume_from ./outputs/deimv2_l/checkpoint-epoch50 --epochs 100

python train_tea.py --pretrained dinov3_s --datasets ./datasets/teabud_dataset_ztu --output_dir ./outputs/deimv2_s --batch_size 4 --lr 0.0001 --train_mode backbone_frozen --unfreeze_backbone_last_n 12 --lr_backbone 0.0000125 --backbone_lr_decay 0.7 --epochs 200 --resume_from ./outputs/deimv2_s/checkpoint-epoch100

python train_tea.py --pretrained dinov3_l --datasets ./datasets/teabud_dataset_ztu ./datasets/TeaLeavesDatasets_split_lr_tea --output_dir ./outputs/deiecececay 0.7 --epochs 200 --resume_from ./outputs/deimv2_l/checkpoint-epoch100

# 多模型 × 多数据集（配置见 eval_config.py 的 DATASETS、DEFAULT_MODELS）
python eval_tea.py
python eval_tea.py --model ./onnx_models/dino_0329_30.onnx ./outputs/deimv2_l/checkpoint-epoch100 --seed 42