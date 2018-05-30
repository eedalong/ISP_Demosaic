export CUDA_VISIABLE_DEVICE=2,4
python evaluate.py --bayer_type='RGGB' --data_dir='/data/dalong/bayer_panasonic/' --flist='/data/dalong/bayer_panasonic/test.txt' --Random=0   --gpu_use=0,1  --checkpoint_folder='../models/DeepISP'  --pad=0 --white_point=65535 --init_model='DemoisaicNet_state_epoch2000.pth' --batchsize=1 --predemosaic=1

