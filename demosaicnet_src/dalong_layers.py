import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import NoiseEstimation as NE
import NoiseEstimation_KPN as NE_KPN
import collections
import config as cfg
CUDA_USE = cfg.CUDA_USE;
class simpleblock(nn.Module):
    def __init__(self,input_channel,output_channel,ksize,pad,batchnorm):
        super(simpleblock,self).__init__();
        if batchnorm:
            self.model = nn.Sequential(
                             nn.Conv2d(input_channel,output_channel,kernel_size = ksize,stride = 1,padding = pad),
                             nn.ReLU(),
                             nn.BatchNorm2d(input_channel)
                             );
        else :
            self.model = nn.Sequential(
                             nn.Conv2d(input_channel,output_channel,kernel_size = ksize,stride = 1,padding = pad),
                             nn.ReLU()
                             );
    def forward(self,inputs):
        outputs = self.model(inputs);
        return outputs;

class lowlevel_block(nn.Module):
    def __init__(self,depth,input_channel,output_channel,ksize,pad=0, batchnorm=True):
        super(lowlevel_block,self).__init__();
        modules = collections.OrderedDict();
        for index in range(depth):
            name = 'conv' + "_{}".format(index+1);
            modules[name] = simpleblock(input_channel,output_channel,ksize,pad,batchnorm);
        self.model = nn.Sequential(modules);

    def forward(self,inputs):
        outputs = self.model(inputs);
        return outputs;

class UnpackBayerMosaicLayer(nn.Module):
    def __init__(self):
        super(UnpackBayerMosaicLayer,self);

    def forward(self,inputs):
        top = Variable(torch.zeros(inputs.size(0),3,inputs.size(2),inputs.size(3)));
        if CUDA_USE:
            top = top.cuda();

        for channel in range(3):
            top[:,channel,::2,::2] = inputs[:,4*c,:,:];
            top[:,channel,::2,1::2] = inputs[:,4*c+1,:,:];
            top[:,channel,1::2,::2] = inputs[:,4*c+2,:,:];
            top[:,channel,1::2,1::2] = inputs[:,4*c+3,:,:];
        return top;
'''
supported bayer type
RGGB GRBG
defalut : GRBG
'''
class BayerMosaicLayer(nn.Module):
    def __init__(self,bayer_type = 'GRBG'):
        super(BayerMosaicLayer,self).__init__();
        self.bayer_type = bayer_type;
    def forward(self,inputs):
        outputs = Variable(torch.zeros(inputs.size(0),3,inputs.size(2),inputs.size(3)));
        if CUDA_USE:
            outputs = outputs.cuda();

        if self.bayer_type == 'GRBG':
            outputs[:,1,::2,::2] = inputs[:,1,::2,::2]; #G
            outputs[:,0,::2,1::2] = inputs[:,0,::2,1::2];# R
            outputs[:,2,1::2,::2] = inputs[:,2,1::2,::2];# B
            outputs[:,1,1::2,1::2] = inputs[:,1,1::2,1::2]; # G
        elif self.bayer_type == 'RGGB' :
            outputs[:,0,::2,::2] = inputs[:,0,::2,::2]; # R
            outputs[:,1,::2,1::2] = inputs[:,1,::2,1::2]; # G
            outputs[:,1,1::2,::2] = inputs[:,1,1::2,::2]; #G
            outputs[:,2,1::2,1::2] = inputs[:,2,1::2,1::2]; # B
        else :
            print('Dude, This bayer type is not supported for now ,sorry for that');
            exit();
        return outputs;
class CropLayer(nn.Module):
    def __init__(self):
        super(CropLayer,self).__init__();

    def forward(self,inputs,reference):
        src_sz = reference.size();
        dst_sz = inputs.size();
        if src_sz == dst_sz :
            return inputs;
        offset = [(s-d) / 2 for s,d in zip(dst_sz,src_sz)];
        outputs = inputs[:,:,int(offset[2]):int(offset[2])+int(src_sz[2]),int(offset[3]):int(offset[3])+int(src_sz[3])].clone();
        return outputs;

class PackBayerMosaicLayer(nn.Module):
    def __init__(self,bayer_type='GRBG' ):
        super(PackBayerMosaicLayer,self).__init__();
        self.bayer_type = bayer_type;
    # receive a raw data
    def forward(self,inputs):
        top = Variable(torch.zeros(inputs.size(0),4,inputs.size(2) / 2,inputs.size(3) / 2));
        if CUDA_USE:
            top = top.cuda();

        if self.bayer_type == 'GRBG':
            '''
            G R G R G
            B G B G B
            G R G R B
            '''
            top[:,0,:,:] = inputs[:,1,::2,::2]; # G
            top[:,1,:,:] = inputs[:,0,::2,1::2]; # R
            top[:,2,:,:] = inputs[:,2,1::2,::2]; # B
            top[:,3,:,:] = inputs[:,1,1::2,1::2]; # G
        if self.bayer_type == 'RGGB':
            '''
            R G R G R
            G B G B G
            R G R G R
            G B G B G
            '''
            top[:,0,:,:] = inputs[:,1,::2,1::2]; # G
            top[:,1,:,:] = inputs[:,0,::2,::2]; # R
            top[:,2,:,:] = inputs[:,2,1::2,1::2]; # B
            top[:,3,:,:] = inputs[:,1,1::2,::2]; # G

        return top;
class UnpackBayerMosaicLayer(nn.Module):
    def __init__(self):
        super(UnpackBayerMosaicLayer,self).__init__();
    def forward(self,inputs):
        outputs = Variable(torch.zeros(inputs.size(0),3,inputs.size(2)*2,inputs.size(3)*2));
        if CUDA_USE:
            outputs = outputs.cuda();

        for channel in range(3):
            outputs[:,channel,::2,::2] = inputs[:,4*channel,:,:];
            outputs[:,channel,::2,1::2] = inputs[:,4*channel+1,:,:];
            outputs[:,channel,1::2,::2] = inputs[:,4*channel+2,:,:];
            outputs[:,channel,1::2,1::2] = inputs[:,4*channel+3,:,:];
        return outputs;
# implement noise estimation and add an noise layer to the inputs
# receive CFA array which has been splited into 4 channels
class AddNoiseEstimationLayer(nn.Module):
    def __init__(self):
        super(AddNoiseEstimationLayer,self).__init__();

    def forward(self,inputs,sigma_info = None):
        if sigma_info is not None :
            outputs = NE_KPN.NoiseEstimation(inputs,sigma_info[:,:,0,0],sigma_info[:,:,0,1]);
            return outputs ;
        else :
            outputs = Variable(torch.rand(inputs.size(0),1,inputs.size(2),inputs.size(3)));
            if CUDA_USE:
                outputs = outputs.cuda();

            for index in range(inputs.size(0)):
                outputs[index,:,:,:] = NE.NoiseEstimation(inputs[index,:,:,:]);
            return outputs;

class SliceHalfLayer(nn.Module):
    def __init__(self):
        super(SliceHalfLayer,self).__init__();

    def forward(self,inputs):
        n = inputs.size(1) / 2;
        A_Half = inputs[:,:n,:,:].clone();
        B_Half = inputs[:,n:,:,:].clone();

        return A_Half*B_Half;

class Upsample_Concat(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Upsample_Concat,self).__init__();
        self.deconv = nn.ConvTranspose2d(in_channels,out_channels,kernel_size =2,stride = 2,bias = False);
    def forward(self,inputs1,inputs2):
        outputs1 = self.deconv(inputs1,output_size = inputs2.size());
        return torch.cat([outputs1,inputs2],1);
class isp_block(nn.Module):
    def __init__(self,input_channel,out_left = 61,out_right = 3):
        super(isp_block,self).__init__();
        self.pad = nn.ReflectionPad2d(1);
        self.block_left = nn.Sequential(
            nn.Conv2d(input_channel,out_left,kernel_size = 3,stride = 1),
            nn.ReLU(),
        );
        self.block_right = nn.Sequential(
            nn.Conv2d(input_channel,out_right,kernel_size = 3,stride = 1),
            nn.ReLU(),
        );

    def forward(self,feats,residual,cat = True):
        if cat :
            feats = torch.cat((feats,residual),1);
        feats = self.pad(feats);
        left_ans = self.block_left(feats);
        right_ans = self.block_right(feats);
        if residual.size()  == right_ans.size():
            residual = residual + right_ans / 10;
        else:
            residual = right_ans;
        return left_ans,right_ans;

class ResnetModule(nn.Module):
    def __init__(self,input_channel,output_channel,Identity_Kernel = 0):
        super(ResnetModule,self).__init__();
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size =3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_value = 0.2)
        );
        if not Identity_Kernel :
            self.block2 = nn.Sequential(
                nn.Conv2d(output_channel ,output_channel,kernel_size = 3, stride = 1,padding =1),
                nn.LeakyReLU(negative_value = 0.2),
                );
        else:
            self.block2 = nn.Sequential(
                nn.Conv2d(output_channel,output_channel,kernel_size =1,padding = 0),
                nn.LeakyReLU(negative_value = 0.2),
            );

    def forward(self,inputs):
        out1 = self.block1(inputs);
        out2 = self.block2(out1);
        return out1+out2;

class DiscriminatorModule(nn.Module):
    def __init__(self,input_channel,output_channel,ksize = 3,padding = 1,stride = 1):
        self.block = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,ksize = ksize,padding = padding = 1,stride = 1 ),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(negative_value = 0.2),
        );

    def forward(self,inputs):

        return self.block(inputs);



