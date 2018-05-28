import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import layer as layers
from torch.nn import init
from collections import OrderedDict
#import init_path
#import caffe

# an pytorch implementation <<Testing procedure>> of Joint Demosaic and Denoising
class DemosaicNet(nn.Module):
    def __init__(self,depth,channel,ksize,pad=0,batchnorm=True,bayer_type = 'GRBG'):
        super(DemosaicNet,self).__init__();
        self.bayer_type = bayer_type
        # Pack the input to 4D inputs
        self.packmosaic = layers.PackBayerMosaicLayer(bayer_type = self.bayer_type);
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

        self.mosaic_mask = layers.BayerMosaicLayer(bayer_type = args.bayer_type);
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
        self.init_with_pretrained();
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

    def forward(self, samples,no_use):
        # 1/4 resolution features
        mosaic = self.mosaic_mask(samples);
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


class DemosaicNetLoss(nn.Module):
    def __init__(self):
        super(DemosaicNetLoss,self).__init__();
        self.pixel_loss = torch.nn.MSELoss();
        self.perceptural_loss = 0;
        self.CropLayer = layers.CropLayer();
    def forward(self,inputs,gt):
        gt = self.CropLayer(gt,inputs)
        loss = self.pixel_loss(inputs,gt);
        return loss;
def Draw_Graph():
    from torchviz import dot
    import pydot

    model = DemosaicNet(15,64,3,1,False,'RGGB');
    exit();
    model = model.cuda();
    model.eval();
    inputs_raw = Variable(torch.rand((1,3,128,128))).cuda();
    inputs_sigma = Variable(torch.rand(1,1,64,64)).cuda();
    outputs = model(inputs_raw,inputs_sigma);

    dot_file = dot.make_dot(outputs,dict(model.named_parameters()));
    dot_file.save('model_arch.dot');
    (graph,) = pydot.graph_from_dot_file('model_arch.dot');
    img_name = '{}.png'.format('model_arch.dot');
    graph.write_png(img_name)
if __name__ == '__main__':
    Draw_Graph();
