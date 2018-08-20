export CUDA_VISIBLE_DEVICES=3

python evaluate_pipe.py \
    --Evaluate=1 \
    --flist='/home/xlyuan/Pipe_Test/test.txt' \
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
    --init_router='DemoisaicNet_state_epoch160.pth' \
    --init_submodel='\
     DemoisaicNet_state_epoch250.pth\
    DemoisaicNet_state_epoch750.pth\
    DemoisaicNet_state_epoch700.pth\
    DemoisaicNet_state_epoch800.pth\
    DemoisaicNet_state_epoch400.pth\
    DemoisaicNet_state_epoch600.pth\
    DemoisaicNet_state_epoch800.pth\
    DemoisaicNet_state_epoch600.pth\
    DemoisaicNet_state_epoch500.pth\
    DemoisaicNet_state_epoch350.pth\
    DemoisaicNet_state_epoch850.pth\
    DemoisaicNet_state_epoch650.pth\
    DemoisaicNet_state_epoch700.pth\
    DemoisaicNet_state_epoch900.pth\
    DemoisaicNet_state_epoch700.pth\
    DemoisaicNet_state_epoch500.pth'\
    --submodel_num=16 \





