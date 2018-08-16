export CUDA_VISIBLE_DEVICES=2

python train.py \
    --print_freq=50 \
    --max_epoch=5000 \
    --flist='/home/xlyuan/ImagesTrain/15/train.txt' \
    --Random=0 \
    --bayer_type='GBRG' \
    --gpu_use=0,1 \
    --model='Submodel' \
    --loss='L1Loss' \
    --TRAIN_BATCH=16 \
    --GET_BATCH=1 \
    --input_type='IMG' \
    --gt_type='IMG' \
    --input_normalize=255 \
    --gt_normalize=255 \
    --checkpoint_folder='./models/SubModel_15' \
    --save_freq=100 \
    --workers=8 \
    --size=128 \
    --pretrained=0 \
    --lr=0.0001 \
    --input_black_point=0 \
    --input_white_point=1 \
    --gt_black_point=0 \
    --gt_white_point=1 \





