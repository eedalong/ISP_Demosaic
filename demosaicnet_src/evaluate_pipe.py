import torch
import time
import dalong_model
import numpy as np
import config as cfg
from torch.autograd import Variable
import os
import shutil
import datasets
import utils
from PIL import Image

#####################################################################
#           WHEN EVALUATING BATCHSIZE WILL ALWAYS BE 1              #
#####################################################################
if os.path.exists('./results/'):
    shutil.rmtree('./results/');
os.makedirs('./results');

correct = 0;
all_samples = 0;
def PSNR(inputs1,inputs2):
    return 10 * np.log10(255.0**2 /(np.mean((inputs1 - inputs2)**2)));
psnr_meter = utils.AverageMeter();

def Test(test_loader,submodels,router):
    global correct , all_samples;
    image_index = 0;
    for model_index in range(len(submodels)):
        submodels[model_index].eval();
    router = router.eval();

    for i ,(inputs,gt) in enumerate(test_loader):
        inputs = Variable(inputs);
        height = inputs.size(2);
        width = inputs.size(3);
        print('dalong log : check inputs size = {}'.format(inputs.size()));
        if cfg.CUDA_USE:
            inputs = inputs.cuda();
        num_width = int(width / 128) ;
        num_height = int(height / 128);
        final_image = np.zeros((3,128 * num_height,128 * num_width));

        for height_index in range(num_height - 1):
            for width_index in range(num_width - 1) :
                inputs_patch = inputs[0,:,height_index * 128: (height_index + 1)* 128,width_index * 128 : (width_index + 1)*128];
                inputs_patch = inputs_patch.unsqueeze(0);
                outputs = router(inputs_patch,0);
                predicted_index = torch.argmax(outputs,dim = 1);
                print('dalong log : predicted index = {}'.format(predicted_index));
                outputs_patch = submodels[predicted_index](inputs_patch,0);
                outputs_patch = outputs_patch.data.cpu().numpy();
                final_image[:,height_index * 128 : (height_index + 1)* 128, width_index * 128 : (width_index + 1)* 128] = np.clip(outputs_patch[0,:,:,:]*255,0,255);
        gt = gt.data.cpu().numpy();
        gt = np.clip(gt[0,:,:num_height * 128 ,:num_width * 128]*255,0,255);
        psnr = PSNR(gt.astype('uint8'),final_image.astype('uint8'));
        final_image = Image.fromarray(final_image.transpose(1,2,0).astype('uint8'));
        final_image.save('./results/image_'+str(image_index)+'.jpg');
        image_index = image_index + 1;
        psnr_meter.update(psnr,1);
        print('dalong log : check psnr =  {}'.format(psnr_meter.value))


def main(args):
    submodels = [];
    for model_index in range(args.submodel_num):
        submodel  = dalong_model.Submodel(args);
        if cfg.CUDA_USE :
            submodel =  torch.nn.DataParallel(submodel);
            submodel = submodel.cuda();
        submodels.append(submodel);

    router = dalong_model.Encoder(args);
    if cfg.CUDA_USE :
        router = router.cuda();
    test_dataset = datasets.dataSet(args);
    test_loader = torch.utils.data.DataLoader(test_dataset,1,shuffle = False,num_workers = int(args.workers),collate_fn = datasets.collate_fn);
    for model_index in range(args.submodel_num):
        init_model = os.path.join('./models/SubModel_'+str(model_index),args.init_submodel[model_index]);
        print('dalong log : for model {} , init with {}'.format(model_index,init_model));
        model_dict = torch.load(init_model);
        submodels[model_index].load_state_dict(model_dict);
    ##############
    # init Router
    #############
    init_model = os.path.join('./models/Encoder',args.init_router);
    model_dict = torch.load(init_model);
    router.load_state_dict(model_dict);

    Test(test_loader,submodels,router);

if __name__ == '__main__':

    parser = cfg.parser;
    args = parser.parse_args();
    tmp = args.init_submodel.split('\\');

    args.init_submodel = [tmp[index].split(' ')[-1] for index in range(1,len(tmp))];
    print('all the params set  = {}'.format(args));
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder);
        print('dalong log : all the models will be saved under {} \n'.format(args.checkpoint_folder));
    utils.save_logs(args.checkpoint_folder,args);
    main(args);



