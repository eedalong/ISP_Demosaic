import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import dalong_layers as layers
from torch.nn import init
from collections import OrderedDict
import config as cfg
#import init_path
#import caffe

# an pytorch implementation <<Testing procedure>> of Joint Demosaic and Denoising
class DemosaicNet(nn.Module):
    def __init__(self,depth,channel,ksize,pad=0,batchnorm=True,bayer_type = 'GRBG'):
        super(DemosaicNet,self).__init__();
        self.bayer_type = bayer_type
        # Pack the input to 4D inputs
        self.packmosaic = layers.PackBayerMosaicLayer();
        # conv_block
        self.preconv = layers.lowlevel_block(1,4,channel,ksize,pad,batchnorm);
        self.block1 = layers.lowlevel_block(depth - 2 ,channel,channel,ksize,pad,batchnorm);
        self.conv15 = layers.lowlevel_block(1,channel,12,ksize ,pad = 0,batchnorm = batchnorm);
        # unpack the residul array to original inputs
        self.unpack = layers.UnpackBayerMosaicLayer();
        # original Mosaic array for 3 channels
        self.mosaic_mask = layers.BayerMosaicLayer(bayer_type = self.bayer_type);
        # crop the input with regard to reference
        self.crop_layer = layers.CropLayer();
        # full resolution convolution
        self.fullres_conv1 = layers.lowlevel_block(1,6,channel,ksize,pad,batchnorm);
        self.fullres_conv2 = nn.Conv2d(channel,3,kernel_size = 1,stride = 1,padding = pad);
        # init the parameters
        self.init_params();

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out');
                if m.bias is not None:
                    init.constant(m.bias, 0);
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1);
                init.constant(m.bias, 0);
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001);
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def init_with_pretrained(self):
        param_name = open('../pretrained/bayer/params/param_name').readlines();
        index  = 0;
        model_parameters = self.parameters();
        for param in param_name :
            tmp_param = np.load('../pretrained/bayer/params/'+param[:-1]+'_conv.npy');
            next(model_parameters).data = torch.Tensor(tmp_param);
            tmp_param = np.load('../pretrained/bayer/params/'+param[:-1]+'_bias.npy');
            next(model_parameters).data = torch.Tensor(tmp_param);

    def forward(self,inputs,sigma):
        '''
        args:
        inputs : bayer array with RGGB or GRBG type
        sigma_info : sigma infomation for noise estimation in dng file
        '''
        inputs_mask = self.mosaic_mask(inputs);

        inputs_pack = self.packmosaic(inputs_mask);

        inputs_conv = self.preconv(inputs_pack);

        inputs_conv = self.block1(inputs_conv);

        inputs_conv = self.conv15(inputs_conv);

        inputs_unpack = self.unpack(inputs_conv);

        cropped_mask = self.crop_layer(inputs_mask,inputs_unpack);

        inputs_fullres = torch.cat((inputs_unpack,cropped_mask),1);

        inputs_fullres = self.fullres_conv1(inputs_fullres);

        inputs_fullres = self.fullres_conv2(inputs_fullres);

        return inputs_fullres;

class BayerNetwork(nn.Module):
    """Released version of the network, best quality.

    This model differs from the published description. It has a mask/filter split
    towards the end of the processing. Masks and filters are multiplied with each
    other. This is not key to performance and can be ignored when training new
    models from scratch.
    """
    def __init__(self,args):
        super(BayerNetwork, self).__init__()

        self.mosaic_mask = layers.BayerMosaicLayer(bayer_type = 'GRBG');
        self.crop_like = layers.CropLayer();
        self.depth = 15
        self.width = args.width
        layers1 = OrderedDict([
            ("pack_mosaic", nn.Conv2d(3, 4, 2, stride=2)),  # Downsample 2x2 to re-establish translation invariance
        ])
        for i in range(self.depth):
            n_out = self.width
            n_in = self.width
            if i == 0:
                n_in = 4
            if i == self.depth-1:
                n_out = 2*self.width
            layers1["conv{}".format(i+1)] = nn.Conv2d(n_in, n_out, 3)
            layers1["relu{}".format(i+1)] = nn.ReLU(inplace=True)

        self.main_processor = nn.Sequential(layers1)
        self.residual_predictor = nn.Conv2d(self.width, 12, 1)
        self.upsampler = nn.ConvTranspose2d(12, 3, 2, stride=2, groups=3)

        self.fullres_processor = nn.Sequential(OrderedDict([
        ("post_conv", nn.Conv2d(6, self.width, 3)),
        ("post_relu", nn.ReLU(inplace=True)),
         ("output", nn.Conv2d(self.width, 3, 1)),
        ]))
        if args.pretrained:
            self.init_with_pretrained();
        else:
            self.init_params();

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out');
                if m.bias is not None:
                    init.constant(m.bias, 0);
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 0);
                init.constant(m.bias, 0);
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001);
                if m.bias is not None:
                    init.constant(m.bias, 0)
    def init_with_pretrained(self):
        print('dalong log : init BayerNet with pretrianed models');
        param_name = open('./pretrained/bayer/param_name').readlines();
        index  = 0;
        model_parameters = self.parameters();
        for param in param_name :
            tmp_param = np.load('./pretrained/bayer/' + param[:-1]);
            print('dalong log : check bayer parameter shape = {}'.format(tmp_param.shape));
            next(model_parameters).data = torch.Tensor(tmp_param);
            #tmp_param = np.load('./pretrained/bayer/' + param[:-1]);
            #next(model_parameters).data = torch.Tensor(tmp_param);

    def forward(self, samples,no_use):
        # 1/4 resolution features
        mosaic = self.mosaic_mask(samples);
        #print(mosaic[0,0,:6,:6]);
        #print(samples[0,0,:6,:6]);
        #print(no_use[0,0,:6,:6]);
        print(mosaic.size(),samples.size(),no_use.size())
        noise  = Variable(torch.FloatTensor(np.zeros((samples.size(0),1,samples.size(2),samples.size(3)))));
        features = self.main_processor(mosaic)
        filters, masks = features[:, :self.width], features[:, self.width:]
        filtered = filters * masks
        residual = self.residual_predictor(filtered)
        upsampled = self.upsampler(residual)

        # crop original mosaic to match output size
        cropped = self.crop_like(mosaic, upsampled)

        # Concated input samples and residual for further filtering
        packed = torch.cat([cropped, upsampled], 1)

        output = self.fullres_processor(packed)

        return output

class UNet2(nn.Module):
    def __init__(self,args):
        super(UNet2,self).__init__();
        self.args = args ;
        self.pack_layer = layers.PackBayerMosaicLayer();
        self.down_layer1 = layers.ResnetModule(4,32);
        self.down_layer2 = layers.ResnetModule(32,64);
        self.down_layer3 = layers.ResnetModule(64,128);
        self.down_layer4 = layers.ResnetModule(128,256);
        self.down_layer5 = layers.ResnetModule(256,512);
        self.up1 = layers.Upsample_Concat(512,256);
        self.up_layer1 = layers.isp_block(512,256 - 3 ,3);
        self.up2 = layers.Upsample_Concat(256,128);
        self.up_layer2 = layers.isp_block(256,128 -3 ,3);
        self.up3 = layers.Upsample_Concat(128,64);
        self.up_layer3 = layers.isp_block(128,64-3,3);
        self.up4 = layers.Upsample_Concat(64,32);
        self.up_layer4 = layers.isp_block(64,32 -3,3);
        self.up_layer5 = layers.isp_block(32,12-3,3);
        self.output_layer = nn.ConvTranspose2d(12,3,2,stride = 2, groups = 3);
        self.init_params();

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out');
                if m.bias is not None:
                    init.constant_(m.bias, 0);
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1);
                init.constant_(m.bias, 0);
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001);
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,inputs,noise):
        inputs = self.pack_layer(inputs);
        down_layer1 = self.down_layer1(inputs);
        down_pool1 = F.max_pool2d(down_layer1,kernel_size = 2,stride = 2);
        down_layer2 = self.down_layer2(down_pool1);
        down_pool2 = F.max_pool2d(down_layer2,kernel_size = 2,stride = 2);
        down_layer3 = self.down_layer3(down_pool2);
        down_pool3 = F.max_pool2d(down_layer3,kernel_size = 2,stride = 2);
        down_layer4 = self.down_layer4(down_pool3);
        down_pool4 = F.max_pool2d(down_layer4,kernel_size = 2,stride = 2);
        down_layer5 = self.down_layer5(down_pool4);
        up1 = self.up1(down_layer5,down_layer4);
        left1,out1 = self.up_layer1(up1,None,cat = False);
        up2 = self.up2(torch.cat((out1,left1),1),down_layer3);
        left2, out2 = self.up_layer2(up2,None,cat = False);
        up3 = self.up3(torch.cat((out2,left2),1),down_layer2);
        left3,out3 = self.up_layer3(up3,None,cat =False);
        up4 = self.up4(torch.cat((out3,left3),1),down_layer1);
        left4,out4 = self.up_layer4(up4,None,cat = False);
        left4,out4 = self.up_layer5(left4,out4,cat = True);
        out5 = self.output_layer(torch.cat((out4,left4),1));
        if self.args.Evaluate :
            return out5;
        return [out5,out4,out3,out2];



class SIDNet(nn.Module):
    def __init__(self,args):
        self.args = args;
        super(SIDNet,self).__init__();
        self.pack_layer = layers.PackBayerMosaicLayer();
        self.down_layer1  = nn.Sequential(
            nn.Conv2d(4,32,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(32,32,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
        );
        self.down_layer2  = nn.Sequential(
            nn.Conv2d(32,64,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(64,64,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
        );
        self.down_layer3  = nn.Sequential(
            nn.Conv2d(64,128,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(128,128,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
        );
        self.down_layer4  = nn.Sequential(
            nn.Conv2d(128,256,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(256,256,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
        );
        self.down_layer5  = nn.Sequential(
            nn.Conv2d(256,512,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
        );
        self.up1 = layers.Upsample_Concat(512,256);
        self.up_layer1  = nn.Sequential(
            nn.Conv2d(512,256,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(256,256,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
        );
        self.up2 = layers.Upsample_Concat(256,128);
        self.up_layer2  = nn.Sequential(
            nn.Conv2d(256,128,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(128,128,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
        );
        self.up3 = layers.Upsample_Concat(128,64);
        self.up_layer3  = nn.Sequential(
            nn.Conv2d(128,64,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(64,64,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
        );

        self.up4 = layers.Upsample_Concat(64,32);
        self.up_layer4  = nn.Sequential(
            nn.Conv2d(64,32,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(32,32,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
        );

        self.up_layer5 = nn.Conv2d(32,12,kernel_size = 1,stride = 1);
        self.output_layer = nn.ConvTranspose2d(12,3,2,stride = 2,groups = 3);
        self.init_params();
        if args.pretrained:
            self.init_with_pretrained();

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out');
                if m.bias is not None:
                    init.constant_(m.bias, 0);
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001);
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def init_with_pretrained(self):
        return ;
        print('dalong log : init with the pretrained params');
        param_name = open('./pretrained/SID/param_name');
        model_parameters = self.parameters();
        param_index = 0;
        root_path = './pretrained/SID/'
        total = len(param_name.readlines());
        while param_index < total:
            weights  = np.load(root_path+str(param_index)+'.npy');
            if len(weights.shape) ==4:
                weights = np.ascontiguousarray(weights.transpose(3,2,0,1));
            print(weights.shape);
            next(model_parameters).data = torch.Tensor(weights);
            param_index = param_index + 1;

    def forward(self,inputs,noise_info):
        inputs = self.pack_layer(inputs);
        down_layer1 = self.down_layer1(inputs);
        down_pool1 = F.max_pool2d(down_layer1,kernel_size = 2,stride = 2);
        down_layer2 = self.down_layer2(down_pool1);
        down_pool2 = F.max_pool2d(down_layer2,kernel_size = 2,stride = 2);
        down_layer3 = self.down_layer3(down_pool2);
        down_pool3 = F.max_pool2d(down_layer3,kernel_size = 2,stride = 2);
        down_layer4 = self.down_layer4(down_pool3);
        down_pool4 = F.max_pool2d(down_layer4,kernel_size = 2,stride = 2);
        down_layer5 = self.down_layer5(down_pool4);
        up1 = self.up1(down_layer5,down_layer4);
        up_layer1 = self.up_layer1(up1);
        up2  = self.up2(up_layer1,down_layer3);
        up_layer2 = self.up_layer2(up2);
        up3 = self.up3(up_layer2,down_layer2);
        up_layer3 = self.up_layer3(up3);
        up4 = self.up4(up_layer3,down_layer1);
        up_layer4 = self.up_layer4(up4);
        up_layer5 = self.up_layer5(up_layer4);
        output = self.output_layer(up_layer5);
        return output;


class UNet(nn.Module):
    '''
    this model integrate resnet module into UNet arch
    '''
    def __init__(self,args,Identical_Kernel = 0):
        super(UNet,self).__init__();
        self.args = args ;
        self.pack_layer = layers.PackBayerMosaicLayer();
        self.down_layer1 = layers.ResnetModule(4,32,Identical_Kernel);
        self.down_layer2 = layers.ResnetModule(32,64,Identical_Kernel);
        self.down_layer3 = layers.ResnetModule(64,128,Identical_Kernel);
        self.down_layer4 = layers.ResnetModule(128,256,Identical_Kernel);
        self.down_layer5 = layers.ResnetModule(256,512,Identical_Kernel);
        self.up1 = layers.Upsample_Concat(512,256);
        self.up_layer1 = layers.ResnetModule(512,256,Identical_Kernel);
        self.up2 = layers.Upsample_Concat(256,128);
        self.up_layer2 = layers.ResnetModule(256,128,Identical_Kernel);
        self.up3 = layers.Upsample_Concat(128,64);
        self.up_layer3 = layers.ResnetModule(128,64,Identical_Kernel);
        self.up4 = layers.Upsample_Concat(64,32);
        self.up_layer4 = layers.ResnetModule(64,32,Identical_Kernel);
        self.up_layer5 = layers.ResnetModule(32,12,Identical_Kernel);
        self.output_layer = nn.ConvTranspose2d(12,3,2,stride = 2, groups = 3);
        self.init_params();

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out');
                if m.bias is not None:
                    init.constant_(m.bias, 0);
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1);
                init.constant_(m.bias, 0);
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001);
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,inputs,noise_info):

        inputs = self.pack_layer(inputs);
        down_layer1 = self.down_layer1(inputs);
        down_pool1 = F.max_pool2d(down_layer1,kernel_size = 2,stride = 2);
        down_layer2 = self.down_layer2(down_pool1);
        down_pool2 = F.max_pool2d(down_layer2,kernel_size = 2,stride = 2);
        down_layer3 = self.down_layer3(down_pool2);
        down_pool3 = F.max_pool2d(down_layer3,kernel_size = 2,stride = 2);
        down_layer4 = self.down_layer4(down_pool3);
        down_pool4 = F.max_pool2d(down_layer4,kernel_size = 2,stride = 2);
        down_layer5 = self.down_layer5(down_pool4);
        up1 = self.up1(down_layer5,down_layer4);
        up_layer1 = self.up_layer1(up1);
        up2  = self.up2(up_layer1,down_layer3);
        up_layer2 = self.up_layer2(up2);
        up3 = self.up3(up_layer2,down_layer2);
        up_layer3 = self.up_layer3(up3);
        up4 = self.up4(up_layer3,down_layer1);
        up_layer4 = self.up_layer4(up4);
        up_layer5 = self.up_layer5(up_layer4);
        output = self.output_layer(up_layer5);
        return output;

class DeNet(nn.Module):
    '''
    this model integrate resnet module into UNet arch
    '''
    def __init__(self,args,Identical_Kernel = 0):
        super(DeNet,self).__init__();
        self.args = args ;
        self.pack_layer = layers.PackBayerMosaicLayer();
        self.down_layer0 = nn.MaxPool2d(kernel_size = self.args.scale_factor,stride = self.args.scale_factor);
        self.down_layer1 = layers.ResnetModule(4,32,Identical_Kernel);
        self.down_layer2 = layers.ResnetModule(32,64,Identical_Kernel);
        self.down_layer3 = layers.ResnetModule(64,128,Identical_Kernel);
        self.down_layer4 = layers.ResnetModule(128,256,Identical_Kernel);
        self.down_layer5 = layers.ResnetModule(256,512,Identical_Kernel);
        self.up1 = layers.Upsample_Concat(512,256);
        self.up_layer1 = layers.ResnetModule(512,256,Identical_Kernel);
        self.up2 = layers.Upsample_Concat(256,128);
        self.up_layer2 = layers.ResnetModule(256,128,Identical_Kernel);
        self.up3 = layers.Upsample_Concat(128,64);
        self.up_layer3 = layers.ResnetModule(128,64,Identical_Kernel);
        self.up4 = layers.Upsample_Concat(64,32);
        self.up_layer4 = layers.ResnetModule(64,32,Identical_Kernel);
        self.up_layer5 = layers.ResnetModule(32,4,Identical_Kernel);
        self.post_conv1 = nn.Conv2d(8,4,kernel_size = 3 , stride = 1,padding = 1);
        self.output_layer = nn.ConvTranspose2d(4,1,2,stride = 2);
        self.up5  = nn.Upsample(scale_factor = self.args.scale_factor,mode = 'bilinear');
        self.init_params();

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out');
                if m.bias is not None:
                    init.constant_(m.bias, 0);
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1);
                init.constant_(m.bias, 0);
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001);
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,inputs,noise_info):
        inputs_pack = self.pack_layer(inputs);
        inputs_downsize = self.down_layer0(inputs_pack);
        down_layer1 = self.down_layer1(inputs_downsize);
        down_pool1 = F.max_pool2d(down_layer1,kernel_size = 2,stride = 2);
        down_layer2 = self.down_layer2(down_pool1);
        down_pool2 = F.max_pool2d(down_layer2,kernel_size = 2,stride = 2);
        down_layer3 = self.down_layer3(down_pool2);
        down_pool3 = F.max_pool2d(down_layer3,kernel_size = 2,stride = 2);
        down_layer4 = self.down_layer4(down_pool3);
        down_pool4 = F.max_pool2d(down_layer4,kernel_size = 2,stride = 2);
        down_layer5 = self.down_layer5(down_pool4);
        up1 = self.up1(down_layer5,down_layer4);
        up_layer1 = self.up_layer1(up1);
        up2  = self.up2(up_layer1,down_layer3);
        up_layer2 = self.up_layer2(up2);
        up3 = self.up3(up_layer2,down_layer2);
        up_layer3 = self.up_layer3(up3);
        up4 = self.up4(up_layer3,down_layer1);
        up_layer4 = self.up_layer4(up4);
        up_layer5 = self.up_layer5(up_layer4);
        up5 = self.up5(up_layer5);
        up5 = torch.cat((up5,inputs_pack),1);
        post_result = self.post_conv1(up5);
        output = self.output_layer(post_result);
        return inputs + output;



class DeepISP(nn.Module):
    def __init__(self,args):
        super(DeepISP,self).__init__();
        self.args = args;
        self.pack = layers.PackBayerMosaicLayer();
        if not self.args.predemosaic:
            self.block1 = layers.isp_block(4);
        else:
            self.block1 = layers.isp_block(3);
        self.block2 = layers.isp_block(64);
        self.block3 = layers.isp_block(64);
        self.block4 = layers.isp_block(64);
        self.block5 = layers.isp_block(64);
        self.block6 = layers.isp_block(64);
        self.block7 = layers.isp_block(64);
        self.block8 = layers.isp_block(64);
        self.block9 = layers.isp_block(64);
        self.block10 = layers.isp_block(64);
        self.block11 = layers.isp_block(64);
        self.block12 = layers.isp_block(64);
        self.block13 = layers.isp_block(64);
        self.block14 = layers.isp_block(64);
        self.block15 = layers.isp_block(64);
        self.block16 = layers.isp_block(64);
        self.block17 = layers.isp_block(64);
        self.block18 = layers.isp_block(64);
        self.block19 = layers.isp_block(64);
        if not self.args.predemosaic:
            self.block20 = nn.Conv2d(64,12,kernel_size = 1,stride = 1);
        else:
            self.block20 = layers.isp_block(64);
        self.unpack = nn.ConvTranspose2d(12,3,2,stride = 2,groups = 3);
    def init_params(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal(m.weight,std = 0.001);
                nn.init.constant(m.bias,0);
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal(m.weight,mode = 'fan_out');
                nn.init.constant(m.bias,0);
    def forward(self,raw,sigma):
        if not self.args.predemosaic:
            raw = self.pack(raw);
        left,right  = self.block1(raw,raw,False);
        left,right = self.block2(left,right);
        left,right = self.block3(left,right);
        left,right = self.block4(left,right);
        left,right = self.block5(left,right);
        left,right = self.block6(left,right);
        left,right = self.block7(left,right);
        left,right = self.block8(left,right);
        left,right = self.block9(left,right);
        left,right = self.block10(left,right);
        left,right = self.block11(left,right);
        left,right = self.block12(left,right);
        left,right = self.block13(left,right);
        left,right = self.block14(left,right);
        left,right = self.block15(left,right);
        left,right = self.block16(left,right);
        left,right = self.block17(left,right);
        left,right = self.block18(left,right);
        left,right = self.block19(left,right);
        if not self.args.predemosaic:
            right =  torch.cat((left,right),1);
            right = self.block20(right);
            right = self.unpack(right);
        else:
            left,right = self.block20(left,right);
        return right;
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__();
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size = 3 , padding = 1,stride =1),
            nn.LeakyReLU(negative_slope = 0.2),
        );
        self.block2 = layers.DiscriminatorModule(64,64);
        self.block3 = layers.DiscriminatorModule(64,128);
        self.block4 = layers.DiscriminatorModule(128,128);
        self.block5 = layers.DiscriminatorModule(128,256);
        self.block6 = layers.DiscriminatorModule(256,256);
        self.block7 = layers.DiscriminatorModule(256,512);
        self.block8 = layers.DiscriminatorModule(512,512);
        self.block9 = nn.Conv2d(512,1,1,stride =1, padding = 0);
        self.AdaptivePool = nn.AdaptiveAvgPool2d(1);
        self.init_params();

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out');
                if m.bias is not None:
                    init.constant_(m.bias, 0);
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1);
                init.constant_(m.bias, 0);
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001);
                if m.bias is not None:
                    init.constant_(m.bias, 0);
    def forward(self,inputs):
        inputs = self.block1(inputs);
        inputs = self.block2(inputs);
        inputs = self.block3(inputs);
        inputs = self.block4(inputs);
        inputs = self.block5(inputs);
        inputs = self.block6(inputs);
        inputs = self.block7(inputs);
        inputs = self.block8(inputs);
        inputs = self.block9(inputs);
        return F.sigmoid(self.AdaptivePool(inputs));

class FilterModel(nn.Module):
    def __init__(self,args):
        super(FilterModel,self).__init__();
        self.args = args ;
        self.model  = nn.Sequential(
            layers.PackBayerMosaicLayer(args.bayer_type),
            nn.Conv2d(4,16,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(16,32,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(32,64,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(64,32,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(32,12,kernel_size = 3,stride = 1,padding = 1),
            nn.ConvTranspose2d(12,3,2,stride = 2),
        );
    def forward(self,inputs,use_less):
        outputs = self.model(inputs);
        return outputs ;



class FastDenoisaicking(nn.Module):
    def __init__(self,args):
        super(FastDenoisaicking,self).__init__();
        self.pack = layers.PackBayerMosaicLayer(args.bayer_type,pack_depth =1)
        self.GMpreprocess = layers.GMPreprocess();
        self.RBpreprocess = layers.RBPreprocess();
        self.Lpreprocess = layers.LPreprocess();
        self.crop = layers.CropLayer()
        self.down_layer1  = nn.Sequential(
            nn.Conv2d(1,16,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(16,32,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
        );
        self.down_layer2  = nn.Sequential(
            nn.Conv2d(32,64,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(64,128,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
        );
        self.down_layer3  = nn.Sequential(
            nn.Conv2d(128,256,kernel_size = 3,stride = 1,padding = 1),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(256,512,kernel_size = 3,stride = 1,padding = 1),
        );
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(512,512),
            nn.LeakyReLU(negative_slope = 0.2),
        );
        self.h2a = nn.Linear(512,23**2);
        self.h2b = nn.Linear(512,23**2);
        self.h1 = nn.Linear(512,23**2);
        self.init_params();
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out');
                if m.bias is not None:
                    init.constant_(m.bias, 0);
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1);
                init.constant_(m.bias, 0);
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001);
                if m.bias is not None:
                    init.constant_(m.bias, 0);
    def PYUV2RGB(self,inputs):
        a = Variable(torch.FloatTensor(np.array([[[[1.0]],[[-1]],[[-2]]],
                                                [[[1.0]],[[-1]],[[0]]],
                                                [[[1.0]],[[-1]],[[2]]]]
                                               )
                                      ),requires_grad = False
                    );
        if cfg.CUDA_USE :
            a = a.cuda();
        return F.conv2d(inputs,a);
    def convolve_per_sample(self,conv_stack,filters):
        results = [];
        for index in range(conv_stack.size(0)):
            results.append(F.conv2d(conv_stack[index,:,:,:].unsqueeze(0),filters[index,:,:,:].unsqueeze(0)));
        results = torch.cat(results,0);
        return results ;
    def forward(self,inputs , useless):
        inputs = self.pack(inputs);
        down_layer1 = self.down_layer1(inputs);
        down_pool1 = F.max_pool2d(down_layer1,kernel_size = 2,stride = 2);
        down_layer2 = self.down_layer2(down_pool1);
        down_pool2 = F.max_pool2d(down_layer2,kernel_size = 2,stride = 2);
        down_layer3 = self.down_layer3(down_pool2);
        fc1_layer = self.pool(down_layer3);
        fc1_layer = fc1_layer.view(-1,512)
        fc2_layer = self.fc1(fc1_layer);
        h1 = self.h1(fc2_layer);
        h2a = self.h2a(fc2_layer);
        h2b = self.h2b(fc2_layer);

        h1 = h1.view((h1.size(0),1,23,23));
        h2a = h2a.view((h2a.size(0),1,23,23));
        h2b = h2b.view((h2b.size(0),1,23,23));

        save_filter1 = h1.data.cpu().numpy();
        np.save('h1',save_filter1);

        save_filter2 = h2a.data.cpu().numpy();
        np.save('h2a',save_filter2);

        save_filter3 = h2b.data.cpu().numpy();
        np.save('h2b',save_filter3);


        fc1m = self.convolve_per_sample(inputs,h1);
        fc1 = self.GMpreprocess(fc1m);
        fc2ma = self.convolve_per_sample(inputs,h2a);
        fc2mb = self.convolve_per_sample(inputs,h2b);
        fc2ma,fc2mb = self.RBpreprocess(fc2ma,fc2mb);
        fc2 = fc2ma - fc2mb ;
        inputs = self.crop(inputs,fc2);
        fl = inputs - fc1m - self.Lpreprocess(fc2);
        output = torch.cat((fl,fc1),1);
        output = torch.cat((output,fc2),1);
        return self.PYUV2RGB(output);
# Encoder model outputs
class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder,self).__init__();
        default_channel = [96,256,384,256];
        kernel_channel = [channel / args.encoder_div  for channel in default_channel];
        print('dalong log : check kernel_chanel = {}'.format(kernel_channel));
        linear = 4 * kernel_channel[-1];
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,kernel_channel[0],kernel_size = 11, stride = 4,padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2,padding = 0),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(kernel_channel[0],kernel_channel[1],kernel_size = 5,stride = 1,padding = 2, groups = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 0 ),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(kernel_channel[1],kernel_channel[2],kernel_size = 3, stride = 1, padding = 1,groups = 1),
            nn.ReLU(),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(kernel_channel[2],kernel_channel[2],kernel_size = 3, stride = 1, padding = 1,groups = 2),
            nn.ReLU(),
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(kernel_channel[2],kernel_channel[1],kernel_size = 3, stride = 1, padding = 1,groups = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2,padding = 0),
        );
        self.fc1 = nn.Sequential(
            nn.Linear(linear,1024),
            nn.ReLU(),
        );
        self.fc2 = nn.Sequential(
            nn.Linear(1024,16),
        )
        self.init_params();
    def init_with_pretrained(self):
        param_name = open('./pretrained/encoder/params.txt');
        param_list = param_name.readlines();
        model_parameters = self.parameters();
        root = './pretrained/encoder/'
        for param in param_list :
            param = root + param[:-1] ;
            weights = np.load(param);
            next(model_parameters).data = torch.Tensor(weights);
        param_name.close();

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out');
                if m.bias is not None:
                    init.constant_(m.bias, 0);
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1);
                init.constant_(m.bias, 0);
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001);
                if m.bias is not None:
                    init.constant_(m.bias, 0);

    def forward(self, inputs,useless ):
        outputs = self.conv1(inputs);
        outputs = self.conv2(outputs);
        outputs = self.conv3(outputs);
        outputs = self.conv4(outputs);
        outputs = self.conv5(outputs);
        outputs = outputs.view((outputs.size(0),-1));
        outputs = self.fc1(outputs);
        outputs = self.fc2(outputs);
        return outputs;

class Submodel(nn.Module):
    def __init__(self,args,depth =  3):
        super(Submodel,self).__init__();
        self.args = args;
        self.depth = depth;
        channel = 64 / args.submodel_div;
        self.BayerMosaic = layers.BayerMosaicLayer(args.bayer_type);
        self.Crop = layers.CropLayer();
        self.layer1 = OrderedDict();
        self.layer1['pack_layer'] = layers.PackBayerMosaicLayer(args.bayer_type);
        for index in range(depth):
            in_channel = 64;
            out_channel = 64;
            if index == 0:
                in_channel = 4;
            if index == depth - 1:
                out_channel = 12;
            self.layer1['layer_{}'.format(index)] = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size = 3),
                nn.LeakyReLU(negative_slope = 0.2),
            );
        self.layer1 = nn.Sequential(self.layer1);
        self.postLayer1 = nn.Sequential(
            nn.ConvTranspose2d(12,3,kernel_size = 2,stride = 2,groups = 3),
            nn.LeakyReLU(negative_slope = 0.2)
        );
        self.postLayer2 = nn.Sequential(
            nn.Conv2d(6,3,kernel_size = 3),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Conv2d(3,3,kernel_size = 3),
        );
        self.init_params();
    def init_params(self):
        for m in self.modules():
            print(m);
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out');
                if m.bias is not None:
                    init.constant_(m.bias, 0);
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1);
                init.constant_(m.bias, 0);
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001);
                if m.bias is not None:
                    init.constant_(m.bias, 0);
    def init_with_pretrained(self):
        model_parameters = self.parameters();
        root = './pretrained/bayer/conv{}_{}.npy';
        for index in range(self.depth):
            param_index = index+1 ;
            if index == self.depth -1 and index !=0 :
                param_index  = 16 ;
            weights = np.load(root.format(param_index,0));
            next(model_parameters).data = torch.Tensor(weights);
            bias = np.load(root.format(param_index,1));
            next(model_parameters).data = torch.Tensor(bias);


    def forward(self,inputs,useless):
        outputs1 = self.layer1(inputs);
        outputs2 = self.postLayer1(outputs1);
        inputs = self.Crop(inputs,outputs2)
        outputs3 = torch.cat((outputs2,inputs),1);
        outputs = self.postLayer2(outputs3);
        return outputs;

if __name__ == '__main__':
    print('Hello World !')
