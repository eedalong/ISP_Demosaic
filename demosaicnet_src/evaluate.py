import torch
import time
import dalong_model
import numpy as np
import config as cfg
from torch.autograd import Variable
import os
import shutil
import datasets
from PIL import Image
import utils
import dalong_loss
if os.path.exists('./results/'):
    shutil.rmtree('./results/');
os.makedirs('./results');
image_index = 0;
psnr_meter = utils.AverageMeter();
ssim_meter = utils.AverageMeter();
Flag = 1;
ssim = dalong_loss.SSIM();
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
    a[image>0.0031308] = (1+0.055)* (a[image>0.0031308]**(1.0/3)) - 0.055;
    a[a<0] = 0;
    a[a>1] = 1;
    return a;

def PSNR(img1,img2,crop,peak_value = 255):
    mse = 0;
    print('dalong : check crop value = {}'.format(crop));
    if crop[0] !=0 and crop[1] !=0:
        mse = np.mean((img1[:,crop[0]:-crop[0],crop[1]:-crop[1]] - img2[:,crop[0]:-crop[0],crop[1]:-crop[1]])**2);
    else:
        print('dalong log : without crop')
        mse = np.mean((img1 - img2)**2);
    return 10*np.log10((peak_value**2)/mse );
def test(train_loader,model):
    global image_index,Flag;
    model.eval();
    start = time.time();
    c = 0;
    crop = 0;
    for i ,(raw,data) in  enumerate(train_loader):
        tmp_start = time.time();
        if not Flag :
            raw_pad = np.zeros((raw.shape[0],raw.shape[1],raw.shape[2]+2*c[0],raw.shape[3]+2*c[1]));
            raw = raw.data.cpu().numpy();
            for index in range(raw.shape[0]):
                raw_pad[index,:,:,:] = np.pad(raw[index,:,:,:],[(0,0),(c[0],c[0]),(c[1],c[1])],'reflect');
            raw = torch.FloatTensor(raw_pad);

        raw_var = Variable(raw.contiguous());
        data = Variable(data);
        if cfg.CUDA_USE :
            raw_var = raw_var.cuda();
            data = data.cuda();
        forward_start = time.time();
        output = model(raw_var,data);
        # JUST FOR DEBUG
        forward_end = time.time();
        print('dalong log : check forward time  = {}s'.format(forward_end - forward_start));
        batchSize = raw_var.size(0);
        output = output.data.cpu().numpy();
        if Flag:
            crop = (np.array(data.shape)[-2:] - np.array(output.shape[-2:])) / 2;
            c = crop;
            print('dalong log : check c  ={}'.format(c));
            Flag = 0;
            continue ;
        data = data.data.cpu().numpy();
        ssim_start = time.time();
        ssim_value = 0;
        #ssim_value = ssim(torch.FloatTensor(output),torch.FloatTensor(data));
        ssim_end = time.time();
        if crop[0] != 0 and crop[1] != 0 :
            pass
        ssim_meter.update(ssim_value,1);

        for index in range(batchSize):

            psnr_start = time.time();
            data_image = (np.clip(output[index,:,:,:] * 255 + 0.5 ,0,255)).astype('uint8')
            save_image = Image.fromarray(data_image.transpose(2,1,0));
            save_image.save('results/image'+str(image_index)+'.jpg');
            input_image = (data[index,:,:,:]*255).astype('uint8');
            psnr = PSNR(data_image,input_image,crop,255);
            input_image = Image.fromarray(input_image.transpose(2,1,0));
            input_image.save('results/input'+str(image_index)+'.jpg');
            psnr_meter.update(psnr);
            tmp_end = time.time();
            image_index = image_index + 1;

        print('dalong log the final psnr value is {}'.format(psnr_meter.value));

def release_memory(model,args):
        pass
def main(args):

    models = {'DemosaicNet':dalong_model.DemosaicNet(args.depth,args.width,args.kernel_size,pad = args.pad,batchnorm = args.batchnorm,bayer_type = args.bayer_type),
              'DeepISP':dalong_model.DeepISP(args),
              'SIDNet':dalong_model.SIDNet(args),
              'BayerNet':dalong_model.BayerNetwork(args),
              'UNet':dalong_model.UNet(args),
              'DeNet':dalong_model.DeNet(args),
              'UNet2':dalong_model.UNet2(args),
              'FastDenoisaicking':dalong_model.FastDenoisaicking(args),
              'FilterModel':dalong_model.FilterModel(args),
              'Submodel':dalong_model.Submodel(args,args.depth),
              }
    test_dataset =  datasets.dataSet(args);
    model = models.get(args.model,'dalong');
    release_memory(models,args);
    if model == 'dalong':
        print('Not A model {}'.format(args.model));
        exit();
    collate_fn = datasets.collate_fn;
    test_loader = torch.utils.data.DataLoader(test_dataset,args.TRAIN_BATCH,shuffle = False,num_workers = int(args.workers),collate_fn = collate_fn);

    print('dalong log : begin to load data');

    init_model = os.path.join(args.checkpoint_folder,args.init_model);
    if cfg.CUDA_USE :
        model = model.cuda();

    if args.init_model != '':
        print('dalong log : init model with {}'.format(args.init_model))
        model_dict = torch.load(init_model);
        model.load_state_dict(model_dict);

    test(test_loader,model);
    print('dalong log : test finished ');

if __name__ == '__main__':

    parser = cfg.parser;
    args = parser.parse_args();
    print('all the params set  = {}'.format(args));
    main(args)
