export CUDA_VISIBLE_DEVICES=1

python train.py \
    --print_freq=50 \
    --max_epoch=5000 \
    --flist='/home/xlyuan/ImagesTrainAll/15/train.txt' \
    --Random=1 \
    --bayer_type='GRBG' \
    --model='Submodel' \
    --loss='L1Loss' \
    --TRAIN_BATCH=16 \
    --GET_BATCH=1 \
    --input_type='IMG' \
    --gt_type='IMG' \
    --input_normalize=255 \
    --gt_normalize=255 \
    --checkpoint_folder='./models/SubModel_15/2' \
    --save_freq=50 \
    --workers=8 \
    --size=64 \
    --pretrained=0 \
    --lr=0.0001 \
    --input_black_point=0 \
    --input_white_point=1 \
    --gt_black_point=0 \
    --gt_white_point=1 \
    --depth=2 \
    --Crop=0 \
    --submodel_div=1 \








