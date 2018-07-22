import torch.utils.data as data
import numpy as np
import os
import torch
import time
import data_reader


def pack_raw(im,args):
    im = (im - args.black_point) / (args.white_point - args.black_point);
    out = np.dstack((im[::2,::2,0],
                    im[::2,1::2,0],
                    im[1::2,::2,0],
                    im[1::2,1::2,0]),
                    );
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
    inputs = batch[0][0];
    gt = batch[0][1]
    for index in range(1,len(batch)):
        inputs = torch.cat((inputs,batch[index][0]),0);
        gt = torch.cat((gt,batch[index][1]),0);
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
        gt_path = paths[1];
        inputs = self.reader.input_loader(input_path);
        gt = self.reader.gt_loader(gt_path);

        inputs = pack_raw(inputs,self.args);
        inputs_final = inputs.transpose(2,0,1);
        inputs_final = np.expand_dims(inputs_final,axis = 0);
        gt_final = gt.transpose(2,0,1);
        gt_final = np.expand_dims(gt_final,axis = 0);

        if self.Random :
            inputs_final = np.zeros((self.args.GET_BATCH ,4,self.args.size,self.args.size));
            gt_final = np.zeros((self.args.GET_BATCH,3,self.args.size*2,self.args.size * 2));
            for read_index in range(self.args.GET_BATCH):
                tmp_input,tmp_gt = self.reader.RandomCrop(self.size,inputs,gt);
                tmp_input,tmp_gt = self.reader.RandomFLipH(tmp_input,tmp_gt);
                tmp_input,tmp_gt = self.reader.RandomFlipV(tmp_input,tmp_gt);
                tmp_input,tmp_gt = self.reader.RandomTranspose(tmp_input,tmp_gt);
                inputs_final[read_index] =tmp_input.transpose(2,0,1).copy();
                gt_final[read_index] = tmp_gt.transpose(2,0,1).copy();

        inputs_final = unpack_raw(inputs_final);
        inputs_final = torch.FloatTensor(inputs_final);
        gt_final = torch.FloatTensor(gt_final);
        data_time_end = time.time();

        return inputs_final,gt_final;

    def __len__(self):
        return len(self.pathlist);

    def get_file_list(self):
        datafile = open(self.flist);
        content = datafile.readlines();
        datafile.close();
        return content ;

