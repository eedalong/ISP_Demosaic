import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import numpy as np
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__();
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,96,kernel_size = 11, stride = 4,padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2,padding = 0),
            nn.LocalResponseNorm(size = 5,alpha = 0.0001,beta = 0.75),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96,256,kernel_size = 5,stride = 1,padding = 2, groups = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 0 ),
            nn.LocalResponseNorm(size = 5, alpha = 0.0001,beta = 0.75),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256,384,kernel_size = 3, stride = 1, padding = 1,groups = 1),
            nn.ReLU(),
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384,384,kernel_size = 3, stride = 1, padding = 1,groups = 2),
            nn.ReLU(),
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384,256,kernel_size = 3, stride = 1, padding = 1,groups = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2,padding = 0),
        );
        self.init_with_pretrained();
    def init_with_pretrained(self):
        model_parameters = self.parameters();
        param_name = open('./models/params.txt');
        param_list = param_name.readlines();
        model_parameters = self.parameters();
        root = './models/'
        for param in param_list :
            param = root + param[:-1] ;
            weights = np.load(param);
            next(model_parameters).data = torch.Tensor(weights);
        param_name.close();

    def forward(self, inputs ):

        outputs = self.conv1(inputs);
        outputs = self.conv2(outputs);
        outputs = self.conv3(outputs);
        outputs = self.conv4(outputs);
        outputs = self.conv5(outputs);
        outputs = outputs.view(-1);
        return outputs


