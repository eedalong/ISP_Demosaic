import torch.utils.data as data
import numpy as np
import os
import torch
import time
import data_reader
from PIL import Image


def collate_fn(batch):
    inputs = batch[0][0];
    gt = batch[0][1];
    noise_std = batch[0][2]
    for index in range(1,len(batch)):
        inputs = torch.cat((inputs,batch[index][0]),0);
        gt = torch.cat((gt,batch[index][1]),0);
        noise_std = torch.cat((noise_std,batch[index][2]),0);
    return inputs,gt,noise_std;

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
        gt_path = paths[1];
        inputs,noise_std = self.reader.input_loader(input_path);
        gt = self.reader.gt_loader(gt_path);

        inputs_final = inputs.transpose(2,0,1);
        inputs_final = np.expand_dims(inputs_final,axis = 0);
        gt_final = gt.transpose(2,0,1);
        gt_final = np.expand_dims(gt_final,axis = 0);

        if self.Random :
            inputs_final = np.zeros((self.args.GET_BATCH ,4,self.args.size,self.args.size));
            if self.args.gt_type == 'DNG_RAW':
                gt_final = np.zeros((self.args.GET_BATCH,4,self.args.size,self.args.size));
            else:
                gt_final = np.zeros((self.args.GET_BATCH,3,self.args.size*2,self.args.size * 2));
            for read_index in range(self.args.GET_BATCH):
                tmp_input,tmp_gt = self.reader.RandomCrop(self.size,inputs,gt);
                tmp_input,tmp_gt = self.reader.RandomFLipH(tmp_input,tmp_gt);
                tmp_input,tmp_gt = self.reader.RandomFlipV(tmp_input,tmp_gt);
                tmp_input,tmp_gt = self.reader.RandomTranspose(tmp_input,tmp_gt);
                inputs_final[read_index] =tmp_input.transpose(2,0,1).copy();
                gt_final[read_index] = tmp_gt.transpose(2,0,1).copy();

        if self.args.gt_type == 'DNG_RAW':
            inputs_final = self.reader.unpack_raw_single(inputs_final);
        else:
            inputs_final = self.reader.unpack_raw(inputs_final);
        gt_final = self.reader.unpack_raw_single(gt_final);

        inputs_final = torch.FloatTensor(inputs_final);
        gt_final = torch.FloatTensor(gt_final);
        data_time_end = time.time();
        return inputs_final,gt_final,noise_std;

    def __len__(self):
        return len(self.pathlist);

    def get_file_list(self):
        datafile = open(self.flist);
        content = datafile.readlines();
        datafile.close();
        return content ;

