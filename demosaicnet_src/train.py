import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import time

import models
import torch.nn as nn
import os
import utils
import datasets as dataset
import argparse
losses = utils.AverageMeter();
recorder = utils.Recorder(save_dir ='./records/');
def train(train_loader,model,criterion,optimizer,epoch,args):
    model.train(True);
    start = time.time();
    for i , (raw,data,sigma) in enumerate(train_loader):
        raw_var = Variable(raw).cuda();
        data_var = Variable(data).cuda();
        sigma = Variable(sigma).cuda();
        out = model(raw,sigma);
        loss = criterion(out,data_var);
        losses.update(loss.data[0],raw.size(0));
        # add to writer

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        end = time.time();
        if i % args.print_freq ==0:
            log_str = 'Epoch:[{0}] [{1} / {2}] \t  Time_Consumed  = {3} , Loss = {4}'.format(epoch,i,len(train_loader),end - start,losses.value);
            print(log_str);
    recorder.add_record('loss',losses.value,epoch);
def main(args):
    train_dataset = dataset.dataSet(args);
    print('dalong log : begin to load data');
    train_loader = torch.utils.data.DataLoader(train_dataset,args.batchsize,shuffle = True,num_workers = int(args.workers));
    #model = models.DemosaicNet(args.depth,args.width,args.kernel_size,pad = args.pad,batchnorm = args.batchnorm,bayer_type = args.bayer_type);
    model = models.BayerNetwork(args);
    print('dalong log : model build finished ');
    criterion = models.DemosaicNetLoss();
    print('dalong log : Loss build finished ');
    model = torch.nn.DataParallel(model,device_ids = list(args.gpu_use));
    model = model.cuda();
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-08);
    for epoch in range(args.max_epoch):
        train(train_loader,model,criterion,optimizer,epoch,args);
        if (epoch +1) % args.save_freq == 0:
            path_checkpoint = '{0}/{1}_state_epoch{2}.pth'.format(args.checkpoint_folder,args.model_name,epoch+1);
            utils.save_checkpoint(model,path_checkpoint);
            print('dalong log : train finished ');
if __name__ == '__main__':
    parser = argparse.ArgumentParser();
    '''
    These are all parameters for training
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
    # if there is a sigma_info file for using ,it is like sigma_shot ,sigma_read
    # if not ,we use our own NoiseEstimation module for noise estimation
    parser.add_argument('--sigma_info',type = bool,default = False,help = 'if this dataset has sigma_info file to use ');
    args = parser.parse_args();
    args.gpu_use = [int(item) for item in list(args.gpu_use[0].split(','))];
    print('all the params set  = {}'.format(args));
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder);
        print('dalong log : all the models will be saved under {} \n'.format(args.checkpoint_folder));
    main(args);






