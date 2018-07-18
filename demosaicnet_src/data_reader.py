from PIL import Image
import numpy as np
import rawpy
import time
import os

class data_reader :
    '''
    This class is intended to provide a unified data reader interface
    for dataset to load different data
    this reader always output 3 channel RGB image numpy array ,scaled to 0-1
    '''
    def __init__(self,input_type = 'IMG',gt_type = 'IMG',args):
        '''
        input_type can be 'IMG','DNG','NUMPY'
        gt_type can be 'IMG','DNG','NUMPY'
        '''
        self.input_type = input_type;
        self.gt_type = gt_type;
        self.args = args;
    def image_loader(self,path):
        img = Image.open(path);
        bitdepth = img.depth;
        img = np.asarray(img);
        if img.shape[2] == 4:
            img = img[:,:,:3];
        return np.float32(img) / np.float32(1<<bitdepth - 1);
    # if the data type is numpy we assume that the data has been preprocessed to be ready for input
    # i.e. the input data has been preprocessed
    def numpy_loader(self,path):
        img = np.load(path);
        if img.shape == 2 :
            img = np.dstack((img,img,img));
        return np.float32(img) ;

    def dng_loader(self,path):
        img = rawpy.imread(path);
        img = img.postprocess(use_camera_wb = True,half_size = False,no_auto_bright = True,output_bps = self.bitdepth);
        img = np.float32(img / np..float32(1<<self.bitdepth - 1 ));
        return img
    def input_loader(self,path):
        if self.input_type == 'IMG':
            return image_loader(path);
        if self.input_type == 'NUMPY':
            return numpy_loader(path);
        if self.input_type == 'DNG':
            return dng_loader(self,path);


    def gt_loader(self,path):
        if self.gt_type == 'IMG':
            return image_loader(path);
        if self.gt_type == 'NUMPY' :
            return numpy_loader(path);
        if self.gt_type == 'DNG':
            return dng_loader(path);

    def RandomCrop(self,size,raw,data):
        h,w = raw.shape[0],raw.shape[1];
        th,tw = size;
        x1 = random.randint(0,w -tw);
        y1 = random.randint(0,h - th);
        if self.gt_type == 'IMG':
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


