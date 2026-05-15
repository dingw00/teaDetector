.venv\Scripts\Activate.ps1

python train_deimv2_s_tealeaves.py --pretrained dinov3_l --datasets ./datasets/teabud_dataset_ztu --output_dir ./outputs/deimv2_l --batch_size 4 --lr 0.0001 --train_mode backbone_frozen --unfreeze_backbone_last_n 12 --lr_backbone 0.0000125 --backbone_lr_decay 0.7 --resume_from ./outputs/deimv2_l/checkpoint-epoch50 --epochs 100

python train_deimv2_s_tealeaves.py --pretrained dinov3_s --datasets ./datasets/teabud_dataset_ztu --output_dir ./outputs/deimv2_s --batch_size 4 --lr 0.0001 --train_mode backbone_frozen --unfreeze_backbone_last_n 12 --lr_backbone 0.0000125 --backbone_lr_decay 0.7 --epochs 200 --resume_from ./outputs/deimv2_s/checkpoint-epoch100

python train_deimv2_s_tealeaves.py --pretrained dinov3_l --datasets ./datasets/teabud_dataset_ztu ./datasets/TeaLeavesDatasets_split_lr_tea --output_dir ./outputs/deiecececay 0.7 --epochs 200 --resume_from ./outputs/deimv2_l/checkpoint-epoch100

python eval_deimv2_tealeaves.py --checkpoint ./outputs/deimv2_l/checkpoint-epoch100 --conf 0.2 --nms 0.3 --batch_size 4 --dataset ./datasets/teabud_dataset_ztu

python eval_onnx_detector.py