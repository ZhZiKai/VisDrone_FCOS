python setup.py build develop



# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=$((RANDOM + 10000)) \
#     tools/train_net.py \
#     --skip-test \
#     --config-file configs/1100_cocopretrained_P2.yaml \
#     DATALOADER.NUM_WORKERS 4 \
#     OUTPUT_DIR training_dir/1100_cocopretrained_P2



CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --skip-test \
    --config-file configs/10012_1100_cocopretrained_2x_P2.yaml \
    DATALOADER.NUM_WORKERS 4 \
    OUTPUT_DIR training_dir/10012_1100_cocopretrained_2x_P2


