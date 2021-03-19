CUDA_VISIBLE_DEVICES=1 \
    python RetinaFace/test_fddb.py \
    --network resnet50 \
    --save_folder RetinaFace/weights/pixta_hard_cases/ \
    --vis_thres 0.3 \
    --save_image \
    --nms_threshold 0.0 \
    --confidence_threshold 0.1 \
    --trained_model RetinaFace/weights/Resnet50_Final.pth \
    #--cpu True
    # --trained_model outputs_all_loss/Resnet18_epoch_95.pth \

    # python full_test.py --cuda 1 --cls_model /home/tungnguyen/Review/cls_models/resnet50mod_25epochs_best_model.pth