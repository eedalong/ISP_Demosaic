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
import time
'''
This is an implementation for dataset.py for Joint Demosaic & Denoising

Datasets are from MSR Dataset and S7-Samsung-image

'''
def pack_raw(raw,args):
    im = raw.raw_image_visible.astype(np.float32);
    im = np.expand_dims(im,axis = 2);
    im = (im - args.black_point) / (args.white_point - args.black_point);
    out = np.concatenate((im[::2,::2,:],
                         im[::2,1::2,:],
                         im[1::2,::2,:],
                         im[1::2,1::2,:]),
                         axis = 2);
    return out ;

def unpack_raw(raw):
    output =  np.zeros((raw.shape[0],3,2*raw.shape[2],2*raw.shape[3]));
    output[:,0,::2,::2] = raw[:,0,:,:];
    output[:,0,::2,1::2] = raw[:,1,:,:];
    output[:,2,1::2,::2] = raw[:,2,:,:];
    output[:,1,1::2,1::2] = raw[:,3,:,:];
    return output ;

def RandomCrop(size,raw,data):
    h,w = raw.shape[0],raw.shape[1];
    th,tw = size;
    x1 = random.randint(0,w -tw);
    y1 = random.randint(0,h - th);
    return raw[y1:y1+th,x1:x1+tw,:],data[y1*2:y1*2+th*2,x1*2:x1*2+tw*2,:];
def RandomFLipH(raw,data):
	if random.random()< 0.5:
		return np.flip(raw,axis = 1).copy(),np.flip(data,axis = 1).copy();
	return raw,data;
def RandomFlipV(raw,data):
 	if random.random()< 0.5:
		return np.flip(raw,axis = 0).copy(),np.flip(data,axis = 0).copy();
	return raw,data;
def RandomTranspose(raw,data):
    if random.random()< 0.5:
        return np.transpose(raw,(1,0,2)).copy(),np.transpose(data,(1,0,2)).copy();
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
        self.inputs = {};
        self.gt = {};
    def __getitem__(self,index):

        data_time_start = time.time();
        '''
        this is for dng and dng training
        dalong : Transfer all the data to the same format and the same
        folder structure
        train.txt  :  input path  target path
        here all raw images are scaled to 0-1 in preprocess processdure
        '''
        paths = self.pathlist[index][:-1].split();
        input_path = os.path.join(self.root,paths[0]);
        gt_path = os.path.join(self.root,paths[1]);
        raw = self.raw_loader(input_path);
        data = self.loader(gt_path);
        raw_inputs = np.zeros((self.args.BATCH ,4,self.args.size,self.args.size));
        data_inputs = np.zeros((self.args.BATCH,3,self.args.size*2,self.args.size * 2));
        if not self.args.FastSID:
            raw  = pack_raw(raw,self.args);
        if self.Random :
            for index in range(self.args.batchsize):
                tmp_raw,tmp_data = RandomCrop(self.size,raw,data);
                tmp_raw,tmp_data = RandomFLipH(tmp_raw,tmp_data);
                tmp_raw,tmp_data = RandomFlipV(tmp_raw,tmp_data);
                tmp_raw,tmp_data = RandomTranspose(tmp_raw,tmp_data);
                raw_inputs[index] =tmp_raw.transpose(2,0,1).copy();
                data_inputs[index] = tmp_data.transpose(2,0,1).copy()

        # raw and data should be scaled to 0-1
        raw_inputs = unpack_raw(raw_inputs);

        raw_inputs = torch.FloatTensor(raw_inputs);
        data_inputs = torch.FloatTensor(data_inputs);
        if self.args.SID:
            in_exposure = gt_exposure = 1;
            if self.args.FastSID:
                in_exposure = float(paths[0][17:-5]);
                gt_exposure = float(paths[1][16:-5]);
            else:
                in_exposure = float(paths[0][22:-5]);
                gt_exposure = float(paths[1][21:-5]);

            ratio = min(gt_exposure / in_exposure,300);
            raw = raw * ratio ;
        sigma = 0;
        data_time_end = time.time();
        #print('dalong log : check data load time = {}s'.format(data_time_end - data_time_start));
        return raw_inputs,data_inputs,sigma;

    def __len__(self):
		return len(self.pathlist);

    def loader(self,filepath):
        global gt;
        if self.args.FastSID:
            output = np.load(filepath).astype('float32') / 65535.0;
            return output;
        else:
            image = rawpy.imread(filepath);
            image = image.postprocess(use_camera_wb = True,half_size = False,no_auto_bright = True,output_bps = 16);
            image = np.float32(image / 65535.0);
            return image;

    def raw_loader(self,filepath):
        global inputs
        if self.args.FastSID:
            output = np.load(filepath);
            return output;
        else:
            return rawpy.imread(filepath);

    def get_file_list(self):
        path = os.path.join(self.root,self.flist);
        datafile = open(path);
        return datafile.readlines();
