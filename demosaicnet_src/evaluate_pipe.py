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
import dalong_loss
from skimage.measure import compare_ssim as ssim
import time
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
ssim_meter = utils.AverageMeter();
index_counter = np.zeros((16,))
def Test(test_loader,submodels,router):
    global correct , all_samples;
    image_index = 0;
    for model_index in range(len(submodels)):
        submodels[model_index].eval();
    router = router.eval();
    total_time = 0;
    for i ,(inputs,gt,noise_map) in enumerate(test_loader):
        inputs = Variable(inputs);
        height = inputs.size(2);
        width = inputs.size(3);
        print('dalong log : check inputs size = {}'.format(inputs.size()));
        num_width = int(width / 120)   ;
        num_height = int(height / 120)  ;
        total_num = num_width * num_height ;
        if cfg.CUDA_USE :
            inputs = inputs.cuda();
        final_image = np.zeros((3,120 * num_height,120 * num_width));
        gt_image = np.zeros((3,120 * num_height,120 * num_width));
        total_time = 0;
        block_index  = 0;
        gt = gt.data.cpu().numpy();
        for height_index in range(num_height ):
            for width_index in range(num_width) :
                block_index = block_index + 1;
                up = max(height_index * 120 -4,0) ;
                up_pad = height_index *120  - up;
                bottom = min((height_index+1) * 120 + 4,height) ;
                bottom_pad = bottom - (height_index+1) *120;
                left = max(width_index * 120 - 4,0) ;
                left_pad = width_index *120  - left;
                right = min((width_index+1) * 120 + 4,width);
                right_pad =  right - (width_index + 1) * 120  ;
                patch_width = right- left ;
                patch_height = bottom - up;
                inputs_patch = inputs[0,:,up:bottom,left : right];
                inputs_patch = inputs_patch.unsqueeze(0);
                start =  time.time();
                outputs = router(inputs_patch,0);
                total_time = total_time + time.time() - start;
                predicted_index = torch.argmax(outputs,dim = 1);
                index_counter[predicted_index] = index_counter[predicted_index] + 1;
                start = time.time();
                outputs_patch = submodels[predicted_index](inputs_patch,0);
                total_time = total_time + time.time() - start;
                outputs_patch = outputs_patch.data.cpu().numpy();

                final_image[:,height_index * 120 : (height_index + 1)* 120, width_index * 120 : (width_index + 1)* 120] = np.clip(outputs_patch[0,:,up_pad:patch_height - bottom_pad,left_pad:patch_width - right_pad]*255,0,255);
                gt_image[:,height_index * 120:(height_index + 1) * 120,width_index * 120 :(width_index + 1)* 120] = np.clip(gt[0,:,height_index * 120:(height_index+1) * 120 ,width_index * 120:(width_index + 1) * 120]*255,0,255);
        gt_image = gt_image.astype('uint8').transpose(1,2,0);
        final_image = final_image.astype('uint8').transpose(1,2,0);
        psnr =  PSNR(gt_image,final_image);
        ssim_value = ssim(gt_image,final_image,multichannel = True);
        input_image = inputs.data.cpu().numpy();
        input_image = (input_image[0,:,:,:]*255).transpose(1,2,0).astype('uint8');
        input_image = Image.fromarray(input_image);
        input_image.save('./results/input_{}.jpg'.format(image_index));
        final_image = Image.fromarray(final_image);
        final_image.save('./results/image_'+str(image_index)+'.jpg');
        gt_image = Image.fromarray(gt_image);
        gt_image.save('./results/gt_'+str(image_index)+'.jpg');
        image_index = image_index + 1;
        psnr_meter.update(psnr,1);
        ssim_meter.update(ssim_value,1);
        print('dalong log : check psnr =  {} ssim = {}'.format(psnr_meter.value,ssim_meter.value));
        print('dalong log : check module choice = {}'.format(index_counter *1.0 / np.sum(index_counter)))
        print('dalong log : check total time and time for each block  = {} {}'.format(total_time,total_time * 1.0 /(num_height * num_width)));


def main(args):
    submodels = [];
    for model_index in range(args.submodel_num):
        submodel  = dalong_model.Submodel(args,args.init_depth[model_index]);
        if cfg.CUDA_USE :
            submodel = submodel.cuda();
        submodels.append(submodel);

    router = dalong_model.Encoder(args);
    if cfg.CUDA_USE :
        router = router.cuda();
    test_dataset = datasets.dataSet(args);
    test_loader = torch.utils.data.DataLoader(test_dataset,1,shuffle = False,num_workers = int(args.workers),collate_fn = datasets.collate_fn);
    for model_index in range(args.submodel_num):
        init_model = os.path.join('./models/SubModel_{}/{}/{}'.format(model_index,args.init_folder[model_index],args.init_submodel[model_index]));
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
    args.init_folder = args.init_folder.split();
    args.init_depth = args.init_depth.split();
    args.init_depth = [int(item) for item in args.init_depth]
    args.init_submodel = [tmp[index].split(' ')[-1] for index in range(1,len(tmp))];
    print('all the params set  = {}'.format(args));
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder);
        print('dalong log : all the models will be saved under {} \n'.format(args.checkpoint_folder));
    utils.save_logs(args.checkpoint_folder,args);
    main(args);



