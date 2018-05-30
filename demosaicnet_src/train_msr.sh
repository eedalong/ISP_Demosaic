export CUDA_VISIABLE_DEVICE=0,1
python train.py --white_point=65535 --bayer_type='RGGB' --data_dir='/data/dalong/bayer_panasonic/' --flist='/data/dalong/bayer_panasonic/train.txt' --Random=0  --predemosaic=1  --gpu_use=0,1  --checkpoint_folder='../models/DeepISP' --pad=0  --pretrained=1 --batchsize=32 --max_epoch=2000 --save_freq=100

