from PIL import Image
import numpy as np
import rawpy
import time
import os
import random
import skimage.transform as sktransform

class data_reader :
    '''
    This class is intended to provide a unified data reader interface
    for dataset to load different data
    this reader always output 3 channel RGB image numpy array ,scaled to 0-1
    '''
    def __init__(self,args,input_type = 'IMG',gt_type = 'IMG'):
        '''
        input_type can be 'IMG','DNG','NUMPY'
        gt_type can be 'IMG','DNG','NUMPY'
        '''
        self.input_type = input_type;
        self.gt_type = gt_type;
        self.args = args;
    def image_loader(self,path,normalize):
        img = Image.open(path);
        img = np.asarray(img);
        if len(img.shape) == 2 :
            img = np.dstack((img,img,img));
        if img.shape[2] == 4:
            img = img[:,:,:3];
        return np.float32(img) / normalize;

    # if the data type is numpy we assume that the data has been preprocessed to be ready for input
    # i.e. the input data has been preprocessed
    def numpy_loader(self,path):
        img = np.load(path);
        if len(img.shape) == 2 :
            img = np.dstack((img,img,img));
        return np.float32(img) ;

    def dngimg_loader(self,path,normalize):
        img = rawpy.imread(path);
        img = img.postprocess(use_camera_wb = True,half_size = False,no_auto_bright = True,output_bps = self.bitdepth);
        img = np.float32(img) / normalize;
        return img
    def dngraw_loader(self,path,normalize):
        # when training to denoise on HDRPlus dataset , we dont do any preprocess
        img = rawpy.imread(path);
        img = img.raw_image_visible;
        img = np.dstack((img,img,img));
        img = np.float32(img) / normalize ;
        return img
    def input_loader(self,path):
        if self.input_type == 'IMG':
            return self.image_loader(path,self.args.input_normalize);
        if self.input_type == 'NUMPY':
            return self.numpy_loader(path);
        if self.input_type == 'DNG_IMG':
            return self.dngimg_loader(path,self.args.input_normalize);
        if self.input_type == 'DNG_RAW':
            out = self.dngraw_loader(path,self.args.input_normalize);
            out = self.pack_raw(out);
            out = self.BLC(out,self.args.input_white_point,self.args.input_black_point);
            out = self.WB(out,self.args.input_white_balance);
            return out;

    def gt_loader(self,path):
        if self.gt_type == 'IMG':
            return self.image_loader(path,self.args.gt_normalize);
        if self.gt_type == 'NUMPY' :
            return self.numpy_loader(path);
        if self.gt_type == 'DNG_IMG':
            return self.dngimg_loader(path,self.args.gt_normalize);
        if self.gt_type == 'DNG_RAW':
            out =self.dngraw_loader(path,self.args.gt_normalize);
            out = self.pack_raw(out);
            out = self.BLC(out,self.args.gt_white_point,self.args.gt_black_point);
            out = self.WB(out,self.args.gt_white_balance);
            return out
    def BLC(self,inputs,white_point,black_point):
        return (inputs - black_point ) / (white_point - black_point);
    def WB(self,inputs,white_balance):
        white_balance = white_balance.split();
        if self.args.bayer_type == 'BGGR':
            inputs[0] = inputs[0] * float(white_balance[2]);
            inputs[1] = inputs[1] * float(white_balance[1]);
            inputs[2] = inputs[2] * float(white_balance[1]);
            inputs[3] = inputs[3] * float(white_balance[0]);
        return inputs;

    def RandomCrop(self,size,raw,data):
        h,w = raw.shape[0],raw.shape[1];
        th,tw = size;
        x1 = random.randint(0,w -tw - 10 );
        y1 = random.randint(0,h - th - 10);
        if self.args.gt_type == 'DNG_RAW':
            return raw[y1:y1+th,x1:x1+tw,:],data[y1:y1+th,x1:x1+tw,:];
        # else we think it is raw --> JPEG
        return raw[y1:y1+th,x1:x1+tw,:],data[y1*2:y1*2+th*2,x1*2:x1*2+tw*2,:];
    def RandomFLipH(self,raw,data):
	    if random.random()< 0.5:
		    return np.flip(raw,axis = 1).copy(),np.flip(data,axis = 1).copy();
	    return raw,data;
    def RandomFlipV(self,raw,data):
 	    if random.random()< 0.5:
		    return np.flip(raw,axis = 0).copy(),np.flip(data,axis = 0).copy();
	    return raw,data;
    def RandomTranspose(self,raw,data):
        if random.random()< 0.5:
            return np.transpose(raw,(1,0,2)).copy(),np.transpose(data,(1,0,2)).copy();
        return raw,data;

    def AddGaussianNoise(self,raw,sigma = 1):

        if self.AddGaussianNoise:
            noise = np.random.normal(loc = 0,scale = sigma,size = raw.shape);
            return raw + noise;
        return raw ;

    def Resize(self,raw,size):

        if self.args.Resize :
            output = sktransform.resize(raw,size);
            return output;
        else:
            return raw ;

    def pack_raw(self,im):
        out = np.dstack((im[::2,::2,1],
                        im[::2,1::2,0],
                        im[1::2,::2,2],
                        im[1::2,1::2,1]),
                        );
        return out ;

    def unpack_raw(self,raw):
        if len(raw.shape) == 3:
            raw = np.expand_dims(raw,axis = 0);
        output =  np.zeros((raw.shape[0],3,2*raw.shape[2],2*raw.shape[3]));
        output[:,1,::2,::2] = raw[:,0,:,:];
        output[:,0,::2,1::2] = raw[:,1,:,:];
        output[:,2,1::2,::2] = raw[:,2,:,:];
        output[:,1,1::2,1::2] = raw[:,3,:,:];
        return output ;

    def unpack_raw_single(self,raw):
        if self.args.gt_type != 'DNG_RAW':
            return raw
        if len(raw.shape) == 3 :
            raw = np.expand_dims(raw,axis = 0);
        output =  np.zeros((raw.shape[0],1,2*raw.shape[2],2*raw.shape[3]));
        output[:,0,::2,::2] = raw[:,0,:,:];
        output[:,0,::2,1::2] = raw[:,1,:,:];
        output[:,0,1::2,::2] = raw[:,2,:,:];
        output[:,0,1::2,1::2] = raw[:,3,:,:];
        return output ;


    def AddPossionNoise(self,raw,sigma = 1):
        pass

