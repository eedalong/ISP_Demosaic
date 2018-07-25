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
        print('dalong log : init the model with pretrianed models');
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
        self.depth = args.depth
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
                init.constant(m.weight, 1);
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
            next(model_parameters).data = torch.Tensor(tmp_param);
            #tmp_param = np.load('./pretrained/bayer/' + param[:-1]);
            #next(model_parameters).data = torch.Tensor(tmp_param);

    def forward(self, samples,no_use):
        # 1/4 resolution features
        mosaic = self.mosaic_mask(samples);
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
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1);
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
        up5 = self.up5(up_layer5)
        output = self.output_layer(up5);
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



def Draw_Graph():
    from torchviz import dot
    import pydot
    args = 1;
    model = SIDNet(args);
    model = model.cuda();
    model.eval();
    layer = layers.CropLayer();

    inputs_raw = Variable(torch.rand((1,3,132,220)));
    inputs_sigma = Variable(torch.rand(1,1,64,64));
    if config.CUDA_USE:
        inputs_raw = inputs_raw.cuda();
        inputs_sigma = inputs_sigma.cuda();
    outputs = model(inputs_raw,inputs_sigma);
    outputs = layer(outputs.outputs);
    print('dalong log : check outputs size = {}'.format(outputs.size()))
    '''
    dot_file = dot.make_dot(outputs,dict(model.named_parameters()));
    dot_file.save('model_arch.dot');
    (graph,) = pydot.graph_from_dot_file('model_arch.dot');
    img_name = '{}.png'.format('model_arch.dot');
    graph.write_png(img_name)
    '''
if __name__ == '__main__':
    Draw_Graph();
