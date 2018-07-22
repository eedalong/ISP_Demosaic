export CUDA_VISIBLE_DEVICES=0,1

python evaluate.py \
    --print_freq=50 \
    --max_epoch=500 \
    --flist='./test.txt' \
    --Random=0 \
    --model='BayerNet' \
    --loss='L1Loss' \
    --TRAIN_BATCH=1 \
    --GET_BATCH=1 \
    --input_type='IMG' \
    --gt_type='IMG' \
    --white_point=65535 \
    --black_point=0 \
    --input_normalize=1 \
    --gt_normalize=255 \
    --checkpoint_folder='./models/UNet_MSR' \
    --save_freq=100 \
    --workers=8 \
    --size=128 \
    --init_model='' \
    --pretrained=1 \


