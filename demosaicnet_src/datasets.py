import torch.utils.data as data
import numpy as np
import os
import torch
import time
import data_reader

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
    if len(raw.shape) == 3:
        raw = np.expand_dims(raw,axis = 0);
    output =  np.zeros((raw.shape[0],3,2*raw.shape[2],2*raw.shape[3]));
    output[:,1,::2,::2] = raw[:,0,:,:];
    output[:,0,::2,1::2] = raw[:,1,:,:];
    output[:,2,1::2,::2] = raw[:,2,:,:];
    output[:,1,1::2,1::2] = raw[:,3,:,:];
    return output ;


def collate_fn(batch):
    raw = batch[0][0];
    data = batch[0][1]
    for index in range(1,len(batch)):
        raw = torch.cat((raw,batch[index][0]),0);
        data = torch.cat((data,batch[index][1]),0);
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
        self.reader = data_reader(input_type = args.input_type, gt_type = args.gt_type,args = self.args);
    def __getitem__(self,index):
        data_time_start = time.time();
        paths =  self.pathlist[index];

        input_path = os.path.join(self.root,paths[0]);
        gt_path = os.path.join(self.root,paths[1]);

        inputs = self.reader.input_loader(input_path);
        gt = self.reader.gt_loader(gt_path);

        inputs_final = inputs.transpose(2,0,1);
        inputs_final = np.expand_dims(inputs_final,axis = 0);
        gt_final = gt.transpose(2,0,1);
        gt_final = np.expand_dims(gt_final,axis = 0);

        if self.Random :
            if self.args.input_type == 'IMG':
                inputs_final = np.zeros((self.args.GET_BATCH ,3,self.args.size,self.args.size));
                gt_final= np.zeros((self.args.GET_BATCH,3,self.args.size,self.args.size));
            else:
                inputs_final = np.zeros((self.args.GET_BATCH ,4,self.args.size,self.args.size));
                gt_final = np.zeros((self.args.GET_BATCH,3,self.args.size*2,self.args.size * 2));
            for read_index in range(self.args.GET_BATCH):
                tmp_input,tmp_gt = self.reader.RandomCrop(self.size,inputs,gt);
                tmp_input,tmp_gt = self.reader.RandomFLipH(tmp_input,tmp_gt);
                tmp_input,tmp_gt = self.reader.RandomFlipV(tmp_input,tmp_gt);
                tmp_input,tmp_gt = self.reader.RandomTranspose(tmp_input,tmp_gt);
                inputs_final[read_index] =tmp_input.transpose(2,0,1).copy();
                gt_final[read_index] = tmp_gt.transpose(2,0,1).copy();
        if self.args.input_type != 'IMG':
            inputs_final = unpack_raw(inputs_final);
        inputs_final = torch.FloatTensor(inputs_final);
        gt_final = torch.FloatTensor(gt_final);
        data_time_end = time.time();
        return inputs_final,gt_final;

    def __len__(self):
        return len(self.pathlist);

    def get_file_list(self):
        path = os.path.join(self.root,self.flist);
        datafile = open(path);
        return datafile.readlines();

