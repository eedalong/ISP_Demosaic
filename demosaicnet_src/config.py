import argparse

CUDA_USE = 1;
parser = argparse.ArgumentParser();
'''
These are all parameters for training and testing
'''

parser.add_argument('--batchsize',type = int,default = 1,help = 'batchsize for training ');
parser.add_argument('--lr',type = float,default = 1e-4,help = 'learning rate for training ');
parser.add_argument('--depth',type = int,default = 15,help = 'number of convolution layers');
parser.add_argument('--kernel_size',type = int,default = 3,help = 'size of kernel for each convolution layer ');
parser.add_argument('--width',type = int,default = 64,help = 'channels for each feature map per layer');
parser.add_argument('--batchnorm',type = bool,default = False,help = 'whether to use batchnorm');
parser.add_argument('--print_freq',type = int ,default = 10,help = 'log print freq when training ');
parser.add_argument('--save_freq',type= int,default =15,help = 'model save freq when training');
parser.add_argument('--workers',type = int,default = 8,help = 'number of workers for reading the data ');
parser.add_argument('--max_epoch',type = int,default = 9,help = 'max epoch for training ');
parser.add_argument('--checkpoint_folder',type = str,default = '../models/');
parser.add_argument('--size',type = int,default = 128,help = 'size for training image setting ');
parser.add_argument('--pad',type = int,default = 0,help = 'paddding size ');
parser.add_argument('--data_dir',type = str,default = '',help = 'dataset directory for training ');
parser.add_argument('--flist',type = str,default = '',help = 'dataset list file for training ');
parser.add_argument('--Center',type = bool,default = False,help = 'whether to crop from center of the image ');
parser.add_argument('--Random',type = int,default = 1,help ='whether to crop randomly from the image ');
parser.add_argument('--gpu_use','--list', nargs='+', help='<Required> Set GPUS to USE', required=False);
parser.add_argument('--model_name',type = str,default = 'DemoisaicNet',help = 'set the model name prefix');
parser.add_argument('--bayer_type',type = str,default = 'GRBG',help = 'set the bayer type for all training data ');
parser.add_argument('--Evaluate',type = bool,default =False,help = 'Whether to evaluate the dataset');
parser.add_argument('--white_point',type = float,default = 255,help  = 'white point for raw data ');
parser.add_argument('--black_point',type = float,default = 0,help = 'black point for raw data ');
parser.add_argument('--pretrained',type = int,default = 0,help = 'whether init the model with pretrained models');
parser.add_argument('--predemosaic',type = int,default = 0,help = ' whether to demosaic the image with biliteral algorithm');
# if there is a sigma_info file for using ,it is like sigma_shot ,sigma_read
# if not ,we use our own NoiseEstimation module for noise estimation
parser.add_argument('--sigma_info',type = bool,default = False,help = 'if this dataset has sigma_info file to use ');
parser.add_argument('--model',type = str,default = 'DemosaicNet',help = 'choose to a Net arch to train ');
parser.add_argument('--loss',type = str,default = '',help = 'choose a loss ')
parser.add_argument('--dataset',type = str,default = '',help  = 'choose a dataset to use ');
parser.add_argument('--SID',type = int,default = 0,help = 'whther to use SID datasets and ratios information');
parser.add_argument('--FastSID',type = int,default = 0,help = 'whether to use the dataset after minimal prerpocess');
parser.add_argument('--TRAIN_BATCH',type = int,default = 4,help = 'train BATCH inputs');
parser.add_argument('--GET_BATCH',type = int,default = 64,help = 'Load GET_BATCH inputs')
parser.add_argument('--lr_change',type = int,default = 1);

