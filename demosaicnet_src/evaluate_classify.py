import torch
import time
import dalong_model
import numpy as np
import config as cfg
from torch.autograd import Variable
import os
import shutil
import datasets_classify as datasets
import utils
correct = 0;
all_samples = 0;
def Test(test_loader,model):
    global correct , all_samples;
    model.eval();
    for i ,(inputs,gt) in enumerate(test_loader):
        inputs = Variable(inputs);
        gt = Variable(gt);
        if cfg.CUDA_USE:
            inputs = inputs.cuda();
            gt = gt.cuda();
        outputs = model(inputs,0);
        batchsize = outputs.size(0);
        outputs = outputs.data.cpu().numpy();
        gt = gt.data.cpu().numpy();

        for index in range(batchsize):
            all_samples = all_samples + 1;
            predicted_index = np.argmax(outputs[index,:]);
            if predicted_index == gt[index] :
                correct = correct + 1;

        print('dalong log : check correctness = {}'.format(correct * 1.0 / all_samples));

def main(args):
    model  = dalong_model.Encoder(args);
    test_dataset = datasets.dataSet(args);
    test_loader = torch.utils.data.DataLoader(test_dataset,1,shuffle = False,num_workers = int(args.workers));
    if cfg.CUDA_USE :
        model = model.cuda();
    init_model = os.path.join(args.checkpoint_folder,args.init_model);
    if args.init_model != '':
        print('dalong log : init model with {}'.format(args.init_model));
        model_dict = torch.load(init_model);
        model.load_state_dict(model_dict);
    Test(test_loader,model);

if __name__ == '__main__':

    parser = cfg.parser;
    args = parser.parse_args();
    print('all the params set  = {}'.format(args));
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder);
        print('dalong log : all the models will be saved under {} \n'.format(args.checkpoint_folder));
    utils.save_logs(args.checkpoint_folder,args);
    main(args);



