import torch.utils.data as data
import numpy as np
import os
import torch
import random
#from torchvision import transform
import NoiseEstimation
import create_CFA
import random
from PIL import Image
import colour_demosaicing.bayer.demosaicing.bilinear   as bilinear
'''
This is an implementation for dataset.py for Joint Demosaic & Denoising

Datasets are from MSR Dataset and S7-Samsung-image

'''

def InverseGamma(image):
    a = image.copy();
    thr = 0.04045;
    alpha = 0.055;
    a[image < thr] = a[image<thr] / 12.92;
    a[image > thr] = ((a[image>thr] + alpha ) / (1 + alpha))**2.4;
    return a;
def AddNoiseEstimation(raw,sigma,args):
	CFA = create_CFA.create_CFA(raw,args.bayer_type);
	sigma_shot = float(sigma.split()[0]);
	sigma_read = float(sigma.split()[1]);
	sigma = np.sqrt(sigma_read **2 + sigma_read * raw);
	return sigma.astype('float');

def AddNoiseEstimation_(raw,args):
	CFA = create_CFA.create_CFA(raw,bayer_type = args.bayer_type);
	sigma_ = NoiseEstimation.NoiseEstimation(CFA);
	sigma	= np.zeros((CFA.shape[0],CFA.shape[1],1));
	sigma[:,:] = sigma_;
	return sigma;

def RandomCrop(size,raw,data):
	w,h = data.size;
	th,tw = size;
	x1 = random.randint(0,w -tw);
	y1 = random.randint(0,h - th);
	return raw.crop((x1,y1,x1+tw,y1+th)),data.crop((x1,y1,x1+tw,y1+th));

def RandomFLipH(raw,data):
	if random.random()< 0.5:
		return raw.transpose(Image.FLIP_LEFT_RIGHT),data.transpose(Image.FLIP_LEFT_RIGHT);
	return raw,data;

def CenterCrop(size,raw,data):
    w,h = data.size;
    th,tw = size;
    x1 = int(round((w - tw) / 2.));
    y1 = int(round((h - th) / 2.));
    return raw.crop((x1,y1,x1+tw,y1+th)),data.crop((x1,y1,x1+tw,y1+th));
class dataSet(data.Dataset):
    def __init__(self,args):
        super(dataSet,self).__init__();
        self.args = args;
        self.root = args.data_dir
        self.flist = args.flist;
        self.pathlist = self.get_file_list();
        self.Random = args.Random;
        self.Evaluate = args.Evaluate;
        self.sigma_info = args.sigma_info;
        self.size = (args.size,args.size);
        if self.sigma_info :
            print('dalong log : check sigma info {}'.format(self.sigma_info));
            sigma_filepth = os.path.join(args.data_dir,'sigma_info');
            self.sigma_file = open(sigma_filepth).readlines();
    def __getitem__(self,index):

        '''
        dalong : Transfer all the data to the same format and the same
        folder structure
        input are in $root/input/*.jpg;
        groundtruths are in $root/groundtruth/*.jpg
        here all raw images are scaled to 0-1 in preprocess processdure
        '''
        input_path = os.path.join(self.root,'input');
        input_path = os.path.join(input_path,self.pathlist[index][:-1]);
        gt_path = os.path.join(self.root,'groundtruth');
        gt_path = os.path.join(gt_path,self.pathlist[index][:-1]);
        raw = self.raw_loader(input_path);
        data = self.loader(gt_path);
        if self.Random :
            raw,data = RandomCrop(self.size,raw,data);
        #raw,data = RandomFLipH(raw,data);
        raw = np.asarray(raw).astype('float32');
        raw = (raw - self.args.black_point*1.0) / (self.args.white_point - self.args.black_point);
        if self.args.predemosaic:
            raw = bilinear.demosaicing_CFA_Bayer_bilinear(raw,self.args.bayer_type);
        if len(raw.shape) == 2:
            raw = np.dstack((raw,raw,raw));
        data = np.asarray(data).astype('float32');
        if data.shape[2] == 4:
            data = data[:,:,:3];
        # raw and data should be scaled to 0-1
        raw = torch.FloatTensor(raw.transpose(2,0,1));
        data = torch.FloatTensor(data.transpose(2,0,1) * 0.00390625);
        sigma_ = 0;
        '''
        if self.sigma_info:
            sigma_ = AddNoiseEstimation(raw,self.sigma_file[index][:-1],self.args);
        else :
            sigma_ = AddNoiseEstimation_(raw,self.args);
        sigma_ = torch.FloatTensor(sigma_.transpose(2,0,1));
        '''
        return raw,data,sigma_;
    def __len__(self):
		return len(self.pathlist);
    def loader(self,filepath):
        image = Image.open(filepath);
        return image;
    def raw_loader(self,filepath):
        return Image.open(filepath);
    def get_file_list(self):
        path = os.path.join(self.root,self.flist);
        datafile = open(path);
        return datafile.readlines();
