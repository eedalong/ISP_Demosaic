export CUDA_VISIBLE_DEVICES=3

python evaluate.py \
    --Evaluate=1 \
    --flist='/home/xlyuan/Pipe_Test/test.txt' \
    --Random=0 \
    --bayer_type='GRBG' \
    --model='BayerNet' \
    --TRAIN_BATCH=1 \
    --GET_BATCH=4 \
    --input_type='IMG' \
    --gt_type='IMG' \
    --input_normalize=255 \
    --gt_normalize=255 \
    --checkpoint_folder='./models/BayerNet/16/' \
    --workers=0 \
    --size=512 \
    --input_black_point=0 \
    --input_white_point=1 \
    --gt_black_point=0 \
    --gt_white_point=1 \
    --pretrained=0 \
    --init_model='DemoisaicNet_state_epoch10.pth' \
    --depth=1 \
    --Crop=1 \
    --demosaicnet_div=16 \








