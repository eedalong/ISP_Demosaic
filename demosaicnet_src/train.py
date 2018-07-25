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

def train(train_loader,model,criterion,optimizer,epoch,args):
    model.train(True);
    start = time.time();
    for i , (inputs,gt) in enumerate(train_loader):
        inputs = Variable(inputs);
        gt = Variable(gt);
        if cfg.CUDA_USE:
            inputs = inputs.cuda();
            gt = gt.cuda();
        out = model(inputs,0);
        loss = criterion(out,gt);
        losses.update(loss.data[0],inputs.size(0));
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        end = time.time();
        if i % args.print_freq ==0:
            log_str = 'Epoch:[{0}] [{1} / {2}] \t  Time_Consumed  = {3} , Loss = {4}'.format(epoch,i,len(train_loader),end - start,losses.value);
            start = time.time();
            print(log_str);

Generator_content_loss = utils.AverageMeter();
Generator_adversarial_loss = utils.AverageMeter();
Generator_total_loss = utils.AverageMeter();
Discriminator_loss = utils.AverageMeter();

def trainGAN(train_loader,generator,discriminator,content_criterion,adversarial_criterion,optim_generator,optim_discriminator,epoch,args):

    ones_const = Variable(torch.ones(args.GET_BATCH* args.TRAIN_BATCH,1)).cuda();
    generator.train(True);
    discriminator.train(True);
    start = time.time();
    for i, (inputs,gt) in enumerate(train_loader):
        inputs = Variable(inputs);
        real = Variable(gt);
        target_real = Variable(torch.rand(args.TRAIN_BATCH * args.GET_BATCH)*0.5 + 0.7);
        target_fake = Variable(torch.rand(args.TRAIN_BATCH * args.GET_BATCH)*0.3);
        if cfg.CUDA_USE :
            inputs = inputs.cuda();
            real = real.cuda();
            target_real = target_real.cuda();
            target_fake = target_fake.cuda();

        fake = generator(inputs,0);
        # Train discriminator
        discriminator.zero_grad();
        discriminator_loss = adversarial_criterion(discriminator(real),target_real) + \
            adversarial_criterion(discriminator(fake),target_fake);
        Discriminator_loss.update(Discriminator_loss.data[0],real.size(0));
        discriminator_loss.backward();
        optim_discriminator.step();
        # Train generator
        generator.zero_grad();
        generator_content_loss = content_criterion(real,fake);
        Generator_content_loss.update(Generator_content_loss.data[0],real.size(0));
        generator_adversarial_loss = adversarial_criterion(discriminator(fake),ones_const);
        Generator_adversarial_loss.update(generator_adversarial_loss.data[0],real.size(0));
        generator_total_loss = generator_content_loss + 1e-3 * Generator_adversarial_loss;
        Generator_total_loss.update(generator_total_loss.data[0],real.size(0));
        Generator_total_loss.backward();
        optim_generator.step();
        end = time.time();
        if i % args.print_freq == 0:
            log_str = 'Epoch:[{0}]  [{1} / {2}] \t Time consumimg  = {3} generator_total_loss = {4} discriminator_loss = {5}'.format(epoch,i,len(train_loader),end - start , Generator_total_loss.value,Discriminator_loss.value);
            print(log_str);

def release_memory(models,losses,args):
    for key in models.keys():
        if key != args.model:
            del models[key];
    for key in losses.keys():
        if key != args.loss:
            del losses[key]
    return ;

def main(args):

    models = {'DemosaicNet':dalong_models.DemosaicNet(args.depth,args.width,args.kernel_size,pad = args.pad,batchnorm = args.batchnorm,bayer_type = args.bayer_type),
              'DeepISP':dalong_models.DeepISP(args),
              'SIDNet':dalong_models.SIDNet(args),
              'BayerNet':dalong_models.BayerNetwork(args),
              'UNet':dalong_models.UNet(args),
              'DeNet':dalong_models.DeNet(args),
              };

    Losses ={'L1Loss':dalong_loss.L1Loss(),
             'L2Loss':dalong_loss.L2Loss(),
             'SSIM':dalong_loss.SSIM(),
             'MSSIM':dalong_loss.MSSSIM(),
             'pixel_perceptural':dalong_loss.pixel_perceptural_loss(),
             'VGGLoss':dalong_loss.VGGLoss(),
             'BCELoss':dalong_loss.BCELoss(),
             };
    release_memory(models,Losses,args);
    train_dataset = datasets.dataSet(args);
    collate_fn = datasets.collate_fn;
    model = models.get(args.model,'dalong');
    train_loader = torch.utils.data.DataLoader(train_dataset,args.TRAIN_BATCH,shuffle = True,collate_fn = datasets.collate_fn,num_workers = int(args.workers));
    criterion  = Losses.get(args.loss,'dalong')
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-08);
    discriminator = None;
    adversarial_criterion = None;
    optim_discriminator = None;
    if args.TRAIN_GAN :
        discriminator = dalong_models.Discriminator();
        discriminator = torch.nn.DataParallel(discriminator);
        adversarial_criterion = dalong_loss.BCELoss();
        optim_discriminator = optim.Adam(discriminator.parameters(),lr = 1e-4,betas = (0.9,0.999),eps = 1e-08,weight_decay =1e-08);
    if cfg.CUDA_USE :
        model = torch.nn.DataParallel(model);
        model = model.cuda();
        if args.TRAIN_GAN :
            discriminator = discriminator.cuda();
    for epoch in range(args.max_epoch):
        if args.TRAIN_GAN :
            trainGAN(train_loader,model,discriminator,criterion,adversarial_criterion,optimizer,optim_discriminator,epoch,args);
        else:
            train(train_loader,model,criterion,optimizer,epoch,args);
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






