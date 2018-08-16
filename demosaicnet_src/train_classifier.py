import torch
import torch.optim as optim
from torch.autograd import Variable
import time
import dalong_model
import torch.nn as nn
import os
import utils
import datasets_classify as datasets
import dalong_loss
import config as cfg
losses = utils.AverageMeter();

def train(train_loader,model,criterion,optimizer,epoch,args):
    model.train(True);
    start = time.time();

    for i , (inputs,gt) in enumerate(train_loader):
        inputs = Variable(inputs);
        gt = Variable(gt);
        if cfg.CUDA_USE :
            inputs = inputs.cuda();
            gt = gt.cuda();
        out = model(inputs,0);
        loss = criterion(out,gt);
        losses.update(loss.item(),inputs.size(0));
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        end = time.time();
        if i % args.print_freq == 0:
            log_str = 'Epoch:[{0}] [{1} / {2}] \t Time_Consumed = {3} , Loss = {4}'.format(epoch,i,len(train_loader),end - start,losses.value);
            print(log_str);


def main(args):
    model = dalong_model.Encoder(args);
    criterion = nn.CrossEntropyLoss();
    train_dataset = datasets.dataSet(args);
    train_loader = torch.utils.data.DataLoader(train_dataset,args.TRAIN_BATCH,shuffle = True,num_workers = int(args.workers));
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr,betas = (0.9,0.999),eps = 1e-8,weight_decay = 1e-8);
    if cfg.CUDA_USE :
        model = model.cuda();

    for epoch in range(args.max_epoch):
        train(train_loader,model,criterion,optimizer,epoch,args);
        if (epoch + 1) % args.save_freq == 0:
            path_checkpoint = '{0}/{1}_state_epoch{2}.pth'.format(args.checkpoint_folder,args.model_name,epoch+1);
            utils.save_checkpoint(model,path_checkpoint);
            print('save model at epoch = {}'.format(epoch + 1));
    print('dalong log : one epoch finished');

if __name__ == '__main__':

    parser = cfg.parser;
    args = parser.parse_args();
    args.gpu_use = [int(item) for item in list(args.gpu_use[0].split(','))];
    print('all the params set  = {}'.format(args));
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder);
        print('dalong log : all the models will be saved under {} \n'.format(args.checkpoint_folder));
    utils.save_logs(args.checkpoint_folder,args);
    main(args);


