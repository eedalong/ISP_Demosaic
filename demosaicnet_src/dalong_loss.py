import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import dalong_layers as layers
import torch.nn.functional as F
import torchvision.models as tmodels
from math import exp

def gaussian(window_size, sigma):
  gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
  return gauss/gauss.sum()

def create_window(window_size, sigma, channel):
  _1D_window = gaussian(window_size, sigma).unsqueeze(1)
  _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
  window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
  return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
  mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
  mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

  mu1_sq = mu1.pow(2)
  mu2_sq = mu2.pow(2)
  mu1_mu2 = mu1*mu2

  sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
  sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
  sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

  C1 = 0.01**2
  C2 = 0.03**2

  ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

  if size_average:
    return ssim_map.mean()
  else:
    return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):
  def __init__(self, window_size = 11, sigma=1.5, size_average = True):
    super(SSIM, self).__init__()
    self.window_size = window_size
    self.size_average = size_average
    self.channel = 1
    self.sigma = sigma
    self.window = create_window(window_size, self.sigma, self.channel)

  def forward(self, img1, img2):
    (_, channel, _, _) = img1.size()

    if channel == self.channel and self.window.data.type() == img1.data.type():
      window = self.window
    else:
      window = create_window(self.window_size, self.sigma, channel)

      if img1.is_cuda:
          window = window.cuda(img1.get_device())
      window = window.type_as(img1)

      self.window = window
      self.channel = channel

    return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class MSSSIM(nn.Module):
  def __init__(self, window_size=11, sigmas=[0.5, 1, 2, 4, 8], size_average = True):
    super(MSSSIM, self).__init__()
    self.SSIMs = [SSIM(window_size, s, size_average=size_average) for s in sigmas]

  def forward(self, img1, img2):
    loss = 1
    for s in self.SSIMs:
      loss *= s(img1, img2)
    return loss


def ssim(img1, img2, window_size = 11, sigma=1.5, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, sigma, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss,self).__init__();
        self.pixel_loss = torch.nn.MSELoss();
        self.perceptural_loss = 0;
        self.CropLayer = layers.CropLayer();
    def forward(self,inputs,gt):
        gt = self.CropLayer(gt,inputs)
        loss = self.pixel_loss(inputs,gt);
        return loss;

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss,self).__init__();
        self.pixel_loss = torch.nn.L1Loss();
        self.perceptural_loss = 0;
        self.CropLayer = layers.CropLayer();
    def forward(self,inputs,gt):
        gt = self.CropLayer(gt,inputs)
        loss = self.pixel_loss(inputs,gt);
        return loss;

class PSNR(nn.Module):
  """ """
  def __init__(self):
    super(PSNR, self).__init__()
    self.mse = nn.MSELoss()
    self.CropLayer = layers.CropLayer();
  def forward(self, target, output):
    target = self.CropLayer(target, output)
    mse = self.mse(output, target)
    return -10 * torch.log(mse) / np.log(10)



class VGGLoss(nn.Module):
  """ """
  def __init__(self, weight=1.0, normalize=True):
    super(VGGLoss, self).__init__()

    self.normalize = normalize;
    self.weight = weight

    vgg = tmodels.vgg16(pretrained=True).features
    slices_idx = [
        [0, 4],
        [4, 9],
        [9, 16],
        [16, 23],
        [23, 30],
        ]
    self.net = torch.nn.Sequential()
    for i, idx in enumerate(slices_idx):
      seq = torch.nn.Sequential()
      for j in range(idx[0], idx[1]):
        seq.add_module(str(j), vgg[j])
      self.net.add_module(str(i), seq)

    for p in self.parameters():
      p.requires_grad = False
    self.net = self.net.cuda();
    self.mse = nn.MSELoss()
    self.CropLayer = layers.CropLayer();

  def forward(self, target, output):
    target = self.CropLayer(target, output)
    output_f = self.get_features(output)
    with torch.no_grad():
      target_f = self.get_features(target)

    losses = []
    for o, t in zip(output_f, target_f):
      losses.append(self.mse(o, t))
    loss = sum(losses)
    if self.weight != 1.0:
      loss = loss * self.weight
    return loss;

  def get_features(self, x):
    """Assumes x in [0, 1]: transform to [-1, 1]."""
    x = 2.0*x - 1.0
    feats = []
    for i, s in enumerate(self.net):
      x = s(x)
      if self.normalize:  # unit L2 norm over features, this implies the loss is a cosine loss in feature space
        f = x / (torch.sqrt(torch.pow(x, 2).sum(1, keepdim=True)) + 1e-8)
      else:
        f = x
      feats.append(f)
    return feats
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss,self).__init__();
        self.bceloss = nn.BCELoss();
    def forward(self,inputs,target):
        return self.bceloss(inputs,target);

class FeatureExtractor_VGG(nn.Module):
    def __init__(self):
        super(FeatureExtractor_VGG,self).__init__();
        # target saves the target input
        self.target = 0;        # mode decides whether we need to do the loss computation
        self.mode = 'none';
        # crit is the crit we use for computing loss
        self.crit = nn.L1Loss();
        self.loss =  0;
        # the model structure is here
        self.features =  nn.Sequential(
            nn.Conv2d(3,64,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64,64,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2,stride = 2,padding = 0,dilation = 1,ceil_mode = False),
            nn.Conv2d(64,128,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128,128,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2,stride = 2,padding = 0,dilation = 1,ceil_mode = False),
            nn.Conv2d(128,256,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256,256,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256,256,kernel_size = 3,stride =1,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2,stride = 2,padding = 0,dilation = 1,ceil_mode = False)
        );
        '''
            nn.Conv2d(256,512,kernel_size = 3,stride = 1,padding= 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512,512,kernel_size = 3,stride = 1 ,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2,stride = 2,padding = 0,dilation = 1,ceil_mode = False),
            nn.Conv2d(512,512,kernel_size = 3,stride = 1, padding =1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512,512,kernel_size = 3,stride = 1, padding =1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512,512,kernel_size = 3,stride = 1, padding =1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2,stride = 2,padding = 0,dilation = 1,ceil_mode = False),
        '''
        self.init();
    def init(self):
        vgg_pretrained = tmodels.vgg16(pretrained = True);
        vgg_params = vgg_pretrained.parameters();
        param_index = 0;
        for param in self.parameters():
            param.data = next(vgg_params).data.clone();
        print('dalong log : module init finished');

    def set_mode(self,mode):
        self.mode = mode;

    def forward(self,inputs):
        output = self.features(inputs);
        if self.mode == 'capture':
            self.target = output.detach();
            #self.target.requires_grad = False;
        elif self.mode == 'loss':
            self.loss = self.crit(output,self.target);
        return output;
class FeatureExtractor_VGG(nn.Module):
    def __init__(self):
        super(FeatureExtractor_VGG,self).__init__();
        # target saves the target input
        self.target = 0;
        # mode decides whether we need to do the loss computation
        self.mode = 'none';
        # crit is the crit we use for computing loss
        self.crit = nn.L1Loss();
        self.loss =  0;
        # the model structure is here
        self.features =  nn.Sequential(
            nn.Conv2d(3,64,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64,64,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2,stride = 2,padding = 0,dilation = 1,ceil_mode = False),
            nn.Conv2d(64,128,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(128,128,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2,stride = 2,padding = 0,dilation = 1,ceil_mode = False),
            nn.Conv2d(128,256,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256,256,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256,256,kernel_size = 3,stride =1,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2,stride = 2,padding = 0,dilation = 1,ceil_mode = False)
        );
        self.features = self.features.cuda();
        '''
            nn.Conv2d(256,512,kernel_size = 3,stride = 1,padding= 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512,512,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512,512,kernel_size = 3,stride = 1 ,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2,stride = 2,padding = 0,dilation = 1,ceil_mode = False),
            nn.Conv2d(512,512,kernel_size = 3,stride = 1, padding =1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512,512,kernel_size = 3,stride = 1, padding =1),
            nn.ReLU(inplace = True),
            nn.Conv2d(512,512,kernel_size = 3,stride = 1, padding =1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2,stride = 2,padding = 0,dilation = 1,ceil_mode = False),
        '''
        self.init();
    def init(self):
        vgg_pretrained = tmodels.vgg16(pretrained = True).cuda();
        vgg_params = vgg_pretrained.parameters();
        param_index = 0;
        for param in self.parameters():
            param.data = next(vgg_params).data.clone();
        print('dalong log : module init finished');

    def set_mode(self,mode):
        self.mode = mode;

    def forward(self,inputs):
        output = self.features(inputs);
        if self.mode == 'capture':
            self.target = output.detach();
            #self.target.requires_grad = False;
        elif self.mode == 'loss':
            self.loss = self.crit(output,self.target);
        return output;

class pixel_perceptural_loss(nn.Module):
    def __init__(self,pixel_weight = 1,percep_weight = 1):
        super(pixel_perceptural_loss,self).__init__();
        self.pixel_weight = pixel_weight;
        self.percep_weight = percep_weight;
        self.pixel = nn.L1Loss();
        self.percep = VGGLoss();
        self.CropLayer = layers.CropLayer();
    def forward(self,outputs,target):

        target = self.CropLayer(target,outputs);
        percep = self.percep(target,outputs)
        pixel = torch.sum(torch.abs(target - outputs)) / (target.size(0)*target.size(1)*target.size(2)*target.size(3));
        print('dalong log : check pixel and percep loss = {} {}'.format(pixel,percep));
        loss = self.pixel_weight*pixel + self.percep_weight * percep;
        #print('dalong log : check loss of two type = {}  {}'.format(pixel,percep));
        return loss ;
