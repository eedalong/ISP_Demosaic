export CUDA_VISIBLE_DEVICES=0,1

python train.py \
    --print_freq=50 \
    --max_epoch=500 \
    --flist='/home/xlyuan/MSR-Demosaicing/Dataset_LINEAR_with_noise/bayer_panasonic/train.txt' \
    --Random=0 \
    --gpu_use=0,1 \
    --model='BayerNet' \
    --loss='L1Loss' \
    --TRAIN_BATCH=16 \
    --GET_BATCH=1 \
    --input_type='IMG' \
    --gt_type='IMG' \
    --input_bitdepth=1 \
    --gt_bitdepth=8 \
    --checkpoint_folder='./models/UNet_MSR' \
    --save_freq=100 \
    --workers=8 \
    --size=64 \
    --black_point=0 \
    --white_point=65535 \


