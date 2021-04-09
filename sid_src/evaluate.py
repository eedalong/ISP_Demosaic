import torch
import time
import models
import torch.nn as nn
import numpy as np
import os
import shutil
import datasets
from torch.autograd import Variable
import argparse
from PIL import Image
import utils
if os.path.exists('./results/'):
    shutil.rmtree('./results/');
os.makedirs('./results');
image_index = 0;
psnr_meter = utils.AverageMeter();

Flag = 1;
def InverseGamma(image):
    a = image.copy();
    thr = 0.04045;
    alpha = 0.055;
    a[image < thr] = a[image<thr] / 12.92;
    a[image > thr] = ((a[image>thr] + alpha ) / (1 + alpha))**2.4;
    return a;

def Gamma(image):
	a = image.copy();
	a [image <= 0.0031308] = a[image <= 0.0031308]*12.92;
	a[image>0.0031308] = (1+0.055)* (a[image>0.0031308]**(1.0/2.4)) - 0.055;
	a[a<0] = 0;
	a[a>1] = 1;
	return a;

def PSNR(img1,img2,crop,peak_value = 255):
    mse = np.mean((img1[:,crop[0]:-crop[0],crop[1]:-crop[1]] - img2[:,crop[0]:-crop[0],crop[1]:-crop[1]])**2);
    return 10*np.log10((peak_value**2)/mse );
def test(train_loader,model):
    global image_index,Flag;
    model.eval();
    start = time.time();
    c = 0;
    crop = 0;
    for i ,(raw,data,sigma_info) in  enumerate(train_loader):
        if not Flag :
            raw_pad = np.zeros((raw.shape[0],raw.shape[1],raw.shape[2]+2*c[0],raw.shape[3]+2*c[1]));
            print('dalong log : check raw size {}'.format(raw.size()));
            raw = raw.data.cpu().numpy();
            for index in range(raw.shape[0]):
                raw_pad[index,:,:,:] = np.pad(raw[index,:,:,:],[(0,0),(c[0],c[0]),(c[1],c[1])],'reflect');
            raw = torch.FloatTensor(raw_pad);
        raw_var = Variable(raw).cuda();
        sigma_info = Variable(sigma_info).cuda();
        output = model(raw_var,sigma_info);
        batchSize = raw_var.size(0);
        output = output.data.cpu().numpy();

        if Flag:
            crop = (np.array(raw.shape)[-2:] - np.array(output.shape[-2:])) / 2;
            c = crop + crop % 2;
            print('dalong log : check c  ={}'.format(c));
            Flag = 0;
            continue ;
        output = output[:,:,int(crop[0]%2):-(int(crop[0]%2)),int(crop[1]%2):-int(crop[1]%2)];
        data = data.data.cpu().numpy();
        for index in range(batchSize):
            data_image = (np.clip(output[index,:,:,:] / 0.00390625 + 0.5 ,0,255)).astype('uint8')
            save_image = Image.fromarray(data_image.transpose(2,1,0));
            save_image.save('image.jpg');
            input_image = (data[index,:,:,:]*255).astype('uint8');
            psnr = PSNR(data_image,input_image,crop);
            input_image = Image.fromarray(input_image.transpose(2,1,0));
            input_image.save('input.jpg');
            psnr_meter.update(psnr);

            #data_image = Image.fromarray(data_image.astype('uint8'));
            #data_image.save('results/'+str(index)+'.jpg');
    print('dalong log : test processdure finished ');
    print('dalong log the final psnr value is {}'.format(psnr_meter.value));
def main(args):

    print('dalong log : begin to load data');
    init_model = os.path.join(args.checkpoint_folder,args.init_model);
    test_dataset = datasets.dataSet(args);
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = args.batchsize,shuffle = False,num_workers = int(args.workers));
    #model = models.DemosaicNet(args.depth,args.width,args.kernel_size,pad = args.pad,batchnorm = args.batchnorm,bayer_type = args.bayer_type);
    model = models.BayerNetwork(args);
    model = torch.nn.DataParallel(model,device_ids = args.gpu_use);
    if args.init_model != '':
        print('dalong log : init model with {}'.format(args.init_model))
        model_dict = torch.load(init_model);
        model.load_state_dict(model_dict);
    model = model.cuda();
    test(test_loader,model);
    print('dalong log : test finished ');

if __name__ == '__main__':
    parser = argparse.ArgumentParser();
    '''
    These are all parameters for training
    '''
    parser.add_argument('--batchsize',type = int,default = 1,help = 'batchsize for training ');
    # model arch is defined here
    parser.add_argument('--depth',type = int,default = 15,help ='number of conv blocks');
    parser.add_argument('--kernel_size',type  = int,default = 3,help = 'size of kernel in the model');
    parser.add_argument('--batchnorm',type = bool,default = False,help = 'whether to use batchnorm in the model ');
    parser.add_argument('--pad',type = int,default = 0,help = 'padding size in the arch');
    parser.add_argument('--width',type = int,default = 64,help = 'channels for each conv layers ');
    parser.add_argument('--workers',type = int,default = 8,help = 'number of workers for reading the data ');
    parser.add_argument('--checkpoint_folder',type = str,default = '../models/');
    parser.add_argument('--size',type = int,default = 128,help = 'size for training image setting ');
    parser.add_argument('--data_dir',type = str,default = '',help = 'dataset directory for training ');
    parser.add_argument('--flist',type = str,default = '',help = 'dataset list file for training ');
    parser.add_argument('--Center',type = bool,default = False,help = 'whether to crop from center of the image ');
    parser.add_argument('--Random',type = int,default = 0,help ='whether to crop randomly from the image ');
    parser.add_argument('--gpu_use','--list', nargs='+', help='<Required> Set GPUS to USE', required=False);
    parser.add_argument('--model_name',type = str,default = 'DemoisaicNet',help = 'set the model name prefix');
    parser.add_argument('--bayer_type',type = str,default = 'GRBG',help = 'set the bayer type for all training data ');
    parser.add_argument('--Evaluate',type = bool,default =False,help = 'Whether to evaluate the dataset');
    # if there is a sigma_info file for using ,it is like sigma_shot ,sigma_read
    # if not ,we use our own NoiseEstimation module for noise estimation
    parser.add_argument('--sigma_info',type = bool,default = False,help = 'if this dataset has sigma_info file to use ');
    parser.add_argument('--white_point',type = float,default = 255,help  = 'white point for raw data ');
    parser.add_argument('--black_point',type = float,default = 0,help = 'black point for raw data ');
    parser.add_argument('--init_model',type = str,default = '',help = 'model name for testing ');
    args = parser.parse_args();
    args.gpu_use = [int(item) for item in list(args.gpu_use[0].split(','))];
    print('all the params set  = {}'.format(args));
    main(args)
