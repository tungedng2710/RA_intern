CUDA_VISIBLE_DEVICES=2 \
    python test_fddb.py \
    --network resnet50 \
    --save_folder weights/pixta_hard_cases/ \
    --vis_thres 0.3 \
    --save_image \
    --nms_threshold 0.0 \
    --confidence_threshold 0.1 \
    --trained_model weights/Resnet50_Final.pth \
    # --trained_model outputs_all_loss/Resnet18_epoch_95.pth \