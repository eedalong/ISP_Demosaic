import torch
import torch.optim as optim
from torch.autograd import Variable
import time
import dalong_models
import torch.nn as nn
import os
import utils
import datasets
import dalong_loss
import config as cfg
losses = utils.AverageMeter();
recorder = utils.Recorder(save_dir ='./records/');
def train(train_loader,model,criterion,optimizer,epoch,args):
    model.train(True);
    start = time.time();
    for i , (raw,data) in enumerate(train_loader):

        raw_var = Variable(raw).cuda();
        data_var = Variable(data).cuda();

        out = model(raw_var,0);


        loss = criterion(out,data_var);


        losses.update(loss.data[0],raw.size(0));
        # add to writer

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        end = time.time();
        if i % args.print_freq ==0:

            log_str = 'Epoch:[{0}] [{1} / {2}] \t  Time_Consumed  = {3} , Loss = {4}'.format(epoch,i,len(train_loader),end - start,losses.value);
            start = time.time();
            print(log_str);
    recorder.add_record('loss',losses.value,epoch);
def release_memory(models,args):
    # TODO
    pass;

def main(args):

    models = {'DemosaicNet':dalong_models.DemosaicNet(args.depth,args.width,args.kernel_size,pad = args.pad,batchnorm = args.batchnorm,bayer_type = args.bayer_type),
              'DeepISP':dalong_models.DeepISP(args),
              'SIDNet':dalong_models.SIDNet(args),
              'BayerNet':dalong_models.BayerNetwork(args)
              }
    Losses ={'L1Loss':dalong_loss.L1Loss(),
             'L2Loss':dalong_loss.L2Loss(),
             'SSIM':dalong_loss.SSIM(),
             'MSSIM':dalong_loss.MSSSIM(),
             'pixel_perceptural':dalong_loss.pixel_perceptural_loss()
             }
    train_dataset = datasets.dataSet(args);
    collate_fn = datasets.collate_fn;
    model = models.get(args.model,'dalong');
    criterion  = Losses.get(args.loss,'dalong')
    release_memory(models,args);
    if model == 'dalong' or criterion == 'dalong' :
        print('Not A model or Loss or Not A Datasets {}  {}'.format(args.model,args.loss));
        exit();
    train_loader = torch.utils.data.DataLoader(train_dataset,args.batchsize,shuffle = True,collate_fn = collate_fn,num_workers = int(args.workers));
    print('dalong log : Loss build finished ');
    model = torch.nn.DataParallel(model);
    model = model.cuda();
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-08);

    for epoch in range(args.max_epoch):
        train(train_loader,model,criterion,optimizer,epoch,args);
        if epoch > 0.5*args.max_epoch and args.lr_change:
            optimizer = torch.optim.Adam(model.parameters(),lr = 1e-5,betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-08);
            args.lr_change = 0;

        if (epoch +1) % args.save_freq == 0:
            path_checkpoint = '{0}/{1}_state_epoch{2}.pth'.format(args.checkpoint_folder,args.model_name,epoch+1);
            utils.save_checkpoint(model,path_checkpoint);
            print('save model at epoch = {}'.format(epoch+1));
    print('dalong log : train finished ');

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






