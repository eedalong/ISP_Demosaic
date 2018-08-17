export CUDA_VISIBLE_DEVICES=3

python evaluate_pipe.py \
    --Evaluate=1 \
    --flist='/home/xlyuan/hdrplus2/test_pipe.txt' \
    --Random=0 \
    --bayer_type='GBRG' \
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
    --init_router='DemoisaicNet_state_epoch1000.pth' \
    --init_submodel='\
     DemoisaicNet_state_epoch5000.pth\
    DemoisaicNet_state_epoch700.pth\
    DemoisaicNet_state_epoch5000.pth\
    DemoisaicNet_state_epoch2500.pth\
    DemoisaicNet_state_epoch5000.pth\
    DemoisaicNet_state_epoch4700.pth\
    DemoisaicNet_state_epoch5000.pth\
    DemoisaicNet_state_epoch5000.pth\
    DemoisaicNet_state_epoch5000.pth\
    DemoisaicNet_state_epoch5000.pth\
    DemoisaicNet_state_epoch5000.pth\
    DemoisaicNet_state_epoch5000.pth\
    DemoisaicNet_state_epoch5000.pth\
    DemoisaicNet_state_epoch5000.pth\
    DemoisaicNet_state_epoch1500.pth\
    DemoisaicNet_state_epoch5000.pth'\
    --submodel_num=16 \





