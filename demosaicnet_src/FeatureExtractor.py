import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
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
        vgg_pretrained = models.vgg16(pretrained = True);
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
def UnitTest():
    extractor1 = FeatureExtractor_VGG();
    L1 = nn.L1Loss();
    inputs1 = Variable(torch.rand((1,3,224,224)),requires_grad = True);
    inputs2 = Variable(1 + torch.rand((1,3,224,224)),requires_grad = True);
    extractor1.set_mode('capture');
    output1 = extractor1(inputs1);
    extractor1.set_mode('loss');
    output2 = extractor1(inputs2);
    output3 = Variable(torch.rand(1,256,28,28));
    print('dalong log : check the oputput shape of output1 and output2 = {}  {}'.format(output1.size(),output2.size()));

    loss = extractor1.loss;
    print('dalong log : check the loss value = {}'.format(loss));
    loss.backward();
if __name__ == '__main__':
    UnitTest();

