import torch.utils.data as data
import numpy as np
import os
import torch
import random
#from torchvision import transform
import NoiseEstimation
import create_CFA
import random
import rawpy
'''
This is an implementation for dataset.py for Joint Demosaic & Denoising

Datasets are from MSR Dataset and S7-Samsung-image

'''
def pack_raw(raw,args):
    im = raw.raw_image_visiable.astype(mp.float32);
    im = np.max((im - args.black_point) / (args.white_point - args.black_point));
    im = np.minimum(im,1.0);
    im = np.expand_dims(im,axis = 2);

    out = np.concatenate(im[::2,::2,:],
                         im[::2,1::2,:],
                         im[1::2,::2,:],
                         im[1::2,1::2,:],
                         axis = 2);
    return out ;


def RandomCrop(size,raw,data):

	h,w = raw.shape[0],raw.shape[1];
	th,tw = size;
	x1 = random.randint(0,w -tw);
	y1 = random.randint(0,h - th);

    return raw[y1:y1+th,x1:x1+tw,:],data[y1*2:y1*2+th*2,x1*2:x1*2+tw*2,:];

def RandomFLipH(raw,data):
	if random.random()< 0.5:
		return np.flip(raw,axis = 1),np.flip(data,axis = 1);
	return raw,data;
def RamdomFlipV(raw,data):
 	if random.random()< 0.5:
		return np.flip(raw,axis = 0),np.flip(data,axis = 0);
	return raw,data;
def RandomTraspose(raw,data):
    if random.random()< 0.5:
        return np.transpose(raw,1,0,2),np.transpose(data,1,0,2);
    return raw,data;

class dataSet(data.Dataset):
    def __init__(self,args):
        super(dataSet,self).__init__();
        self.args = args;
        self.root = args.data_dir
        self.flist = args.flist;
        self.pathlist = self.get_file_list();
        self.Random = args.Random;
        self.Evaluate = args.Evaluate;
        self.size = (args.size,args.size);
    def __getitem__(self,index):

        '''
        this is for dng and dng training
        dalong : Transfer all the data to the same format and the same
        folder structure
        input are in $root/input/*.dng;
        groundtruths are in $root/groundtruth/*.dng
        here all raw images are scaled to 0-1 in preprocess processdure
        '''
        input_path = os.path.join(self.root,'input');
        input_path = os.path.join(input_path,self.pathlist[index][:-1]);
        gt_path = os.path.join(self.root,'groundtruth');
        gt_path = os.path.join(gt_path,self.pathlist[index][:-1]);
        raw = self.raw_loader(input_path);
        data = self.loader(gt_path);
        raw  = pack_raw(raw,self.args);
        if self.Random :
            raw,data = RandomCrop(self.size,raw,data);
            raw,data = RandomFLipH(raw,data);
            raw,data = RandomFlipV(raw,data);
            raw,data = RandomTranspose(raw,data);
        # raw and data should be scaled to 0-1
        raw = torch.FloatTensor(raw.transpose(2,0,1));
        data = torch.FloatTensor(data.transpose(2,0,1));
        return raw,data;

    def __len__(self):
		return len(self.pathlist);

    def loader(self,filepath):
        image = rawpy.imread(filepath);
        image = image.postprocess(use_camera_wb = True,half_size = False,no_auto_bright = True,output_bps = 16);
        image = np.float32(image / 65535.0);
        return image;

    def raw_loader(self,filepath):
        return rawpy.imread(filepath);

    def get_file_list(self):
        path = os.path.join(self.root,self.flist);
        datafile = open(path);
        return datafile.readlines();
