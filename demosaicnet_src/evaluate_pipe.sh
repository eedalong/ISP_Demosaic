export CUDA_VISIBLE_DEVICES=3

python evaluate_pipe.py \
    --Evaluate=1 \
    --flist='/home/xlyuan/Pipe_Test/test.txt' \
    --Random=1 \
    --bayer_type='GRBG' \
    --GET_BATCH=1 \
    --input_type='IMG' \
    --gt_type='IMG' \
    --input_normalize=255 \
    --gt_normalize=255 \
    --workers=0 \
    --input_black_point=0 \
    --input_white_point=1 \
    --gt_black_point=0 \
    --gt_white_point=1 \
    --pretrained=0 \
    --init_router='8/DemoisaicNet_state_epoch20.pth' \
    --init_folder='1/ 5/16 5/16 1/ 1/ 1/ 1/ 1/ 5/16 5/16 1/ 1/ 5/16 5/16 5/16 5/16' \
    --init_depth='1 5 5 1 1 1 1 1 5 5 1 1 5 5 5 5' \
    --init_submodel='\
     DemoisaicNet_state_epoch50.pth\
    DemoisaicNet_state_epoch450.pth\
    DemoisaicNet_state_epoch500.pth\
    DemoisaicNet_state_epoch150.pth\
    DemoisaicNet_state_epoch50.pth\
    DemoisaicNet_state_epoch50.pth\
    DemoisaicNet_state_epoch50.pth\
    DemoisaicNet_state_epoch100.pth\
    DemoisaicNet_state_epoch300.pth\
    DemoisaicNet_state_epoch50.pth\
    DemoisaicNet_state_epoch200.pth\
    DemoisaicNet_state_epoch200.pth\
    DemoisaicNet_state_epoch400.pth\
    DemoisaicNet_state_epoch500.pth\
    DemoisaicNet_state_epoch400.pth\
    DemoisaicNet_state_epoch300.pth'\
    --submodel_num=16 \
    --depth=1 \
    --encoder_div=8 \
    --real_patchsize=120 \
    --test_patchsize=128 \
    --size=512 \
    --Crop=1 \








