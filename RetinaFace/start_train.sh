    # --training_dataset ./data/widerface/train/label.txt \
CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --training_dataset ./final_data.txt \
    --val_dataset ./data/widerface/val/label.txt \
    --network resnet50 \
    --num_workers 8 \
    --lr 1e-3 \
    --momentum 0.9 \
    --resume_epoch 0 \
    --weight_decay 5e-4 \
    --gamma 0.1 \
    --save_folder outputs_data_pixta \
    --resume_net weights/Resnet50_Final.pth \