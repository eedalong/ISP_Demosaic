import torch.utils.data as data
import numpy as np
import os
import torch
import time
import data_reader
from PIL import Image
import random

def collate_fn(batch):
    inputs = batch[0][0];
    gt = batch[0][1]
    for index in range(1,len(batch)):
        inputs = torch.cat((inputs,batch[index][0]),0);
        gt = torch.cat((gt,batch[index][1]),0);
    return inputs,gt;
def unpack_raw(raw,args):

    output =  np.zeros((3,2*raw.shape[1],2*raw.shape[2]));
    if args.bayer_type == 'RGGB':
        output[0,::2,::2] = raw[0,:,:];
        output[1,::2,1::2] = raw[1,:,:];
        output[1,1::2,::2] = raw[2,:,:];
        output[2,1::2,1::2] = raw[3,:,:];
    if args.bayer_type == 'GBRG':
        output[1,::2,::2] = raw[1,:,:];
        output[2,::2,1::2] = raw[2,:,:];
        output[0,1::2,::2] = raw[0,:,:];
        output[1,1::2,1::2] = raw[1,:,:];
    if args.bayer_type == 'GRBG':
        output[1,::2,::2] = raw[1,:,:];
        output[0,::2,1::2] = raw[0,:,:];
        output[2,1::2,::2] = raw[2,:,:];
        output[1,1::2,1::2] = raw[1,:,:];

    return output ;
def RandomFLipH(raw):
	if random.random()< 0.5:
		return np.flip(raw,axis = 1).copy();
	return raw;
def RandomFlipV(raw):
 	if random.random()< 0.5:
		return np.flip(raw,axis = 0).copy();
	return raw;
def RandomTranspose(raw):
    if random.random()< 0.5:
        return np.transpose(raw,(1,0,2)).copy();
    return raw;

def collate_fn(batch):
    inputs = batch[0][0];
    gt = batch[0][1]
    for index in range(1,len(batch)):
        inputs = torch.cat((inputs,batch[index][0]),0);
        gt = torch.cat((gt,batch[index]),0);
    return inputs,gt;



class dataSet(data.Dataset):
    def __init__(self,args):
        super(dataSet,self).__init__();
        self.args = args;
        self.flist = args.flist;
        self.pathlist = self.get_file_list();
        self.Random = args.Random;
        self.Evaluate = args.Evaluate;
        self.size = (args.size,args.size);
        self.reader = data_reader.data_reader(args,input_type = args.input_type, gt_type = args.gt_type);
    def __getitem__(self,index):
        data_time_start = time.time();
        paths =  self.pathlist[index][:-1].split();
        input_path = paths[0];
        gt = int(paths[1]);
        inputs = self.reader.input_loader(input_path);

        if self.Random:
            tmp_inputs = RandomFLipH(inputs);
            tmp_inputs = RandomFlipV(tmp_inputs);
            tmp_inputs = RandomTranspose(tmp_inputs);
        tmp_inputs = tmp_inputs.transpose(2,0,1);
        inputs = unpack_raw(tmp_inputs,self.args);
        inputs = torch.FloatTensor(inputs);
        data_time_end = time.time();
        return inputs,gt;

    def __len__(self):
        return len(self.pathlist);

    def get_file_list(self):
        datafile = open(self.flist);
        content = datafile.readlines();
        datafile.close();
        return content ;

