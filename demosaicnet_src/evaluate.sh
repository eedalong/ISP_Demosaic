export CUDA_VISIBLE_DEVICES=3

python evaluate.py \
    --Evaluate=1 \
    --flist='/home/xlyuan/ImagesTrainAll/2/test.txt' \
    --Random=0 \
    --bayer_type='GBRG' \
    --model='Submodel' \
    --TRAIN_BATCH=1 \
    --GET_BATCH=1 \
    --input_type='IMG' \
    --gt_type='IMG' \
    --input_normalize=255 \
    --gt_normalize=255 \
    --checkpoint_folder='./SubModel_2/6/' \
    --workers=0 \
    --size=24 \
    --input_black_point=0 \
    --input_white_point=1 \
    --gt_black_point=0 \
    --gt_white_point=1 \
    --pretrained=0 \
    --init_model='DemoisaicNet_state_epoch200.pth' \
    --depth=6 \







