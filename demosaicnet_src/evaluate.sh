export CUDA_VISIBLE_DEVICES=3

python evaluate.py \
    --Evaluate=1 \
    --flist='/home/xlyuan/MSR/Dataset_LINEAR_without_noise/bayer_panasonic/test.txt' \
    --Random=0 \
    --bayer_type='GRBG' \
    --model='BayerNet' \
    --TRAIN_BATCH=1 \
    --GET_BATCH=1 \
    --input_type='IMG' \
    --gt_type='IMG' \
    --input_normalize=255 \
    --gt_normalize=255 \
    --checkpoint_folder='./models/BayerNet' \
    --workers=0 \
    --size=64 \
    --input_black_point=0 \
    --input_white_point=1 \
    --gt_black_point=0 \
    --gt_white_point=1 \
    --pretrained=1 \
    --init_model='' \
    --depth=6 \
    --Crop=0 \







