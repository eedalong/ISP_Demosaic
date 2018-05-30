import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import dalong_layers as layers
import config
class SIDNet(nn.Module):
    def __init__(self):
        super(SIDNet,self).__init__();
        self.pack_mosaic = layers.PackBayerMosaicLayer(bayer_type = 'GRBG');
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

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
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

    def forward(self,inputs,noise_info):
        packed_input  = self.pack_mosaic(inputs);
        down_layer1 = self.down_layer1(packed_input);
        print('dalong log : check down_layer1 size  = {}'.format(down_layer1.size()));
        down_pool1 = F.max_pool2d(down_layer1,kernel_size = 2,stride = 2);

        print('dalong log : check down_pool1 size  = {}'.format(down_pool1.size()));
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
        print('dalong log : check output size = {}'.format(output.size()));
        return output;
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

def Draw_Graph():
    from torchviz import dot
    import pydot

    model = SIDNet();
    if config.CUDA_USE:
        model = model.cuda();
    model.eval();
    inputs_raw = Variable(torch.rand((1,3,132,220)));
    inputs_sigma = Variable(torch.rand(1,1,64,64));
    if config.CUDA_USE:
        inputs_raw = inputs_raw.cuda();
        inputs_sigma = inputs_sigma.cuda();
    outputs = model(inputs_raw,inputs_sigma);
    dot_file = dot.make_dot(outputs,dict(model.named_parameters()));
    dot_file.save('model_arch.dot');
    (graph,) = pydot.graph_from_dot_file('model_arch.dot');
    img_name = '{}.png'.format('model_arch.dot');
    graph.write_png(img_name)
if __name__ == '__main__':
    Draw_Graph();
