import numpy as np
import rawpy
import time
import os.path  as osp

def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:],
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out


input_file = open('/data/dalong/Sony/Sony_train_list.txt');
root = '/data/dalong/Sony/'
save_root = '/data/dalong/Sony/SID/'
save_file = open('Sony_test_file.txt','w');
'''
do the post process to gt dng
do the minimal preprocess to input_dng
'''

for item in input_file.readlines():
    path = item[:-1].split();
    start = time.time();
    print(osp.join(root,path[0]));
    input_raw = rawpy.imread(osp.join(root,path[0]));
    input_raw = pack_raw(input_raw);
    input_filename = osp.basename(path[0][:-4]);

    gt_filename = osp.basename(path[1][:-4]);
    print('dalong log : check input file name = {}'.format(input_filename));
    gt_raw = rawpy.imread(osp.join(root,path[1]));
    gt_image = gt_raw.postprocess(use_camera_wb = True,half_size = False,no_auto_bright = True,output_bps =16);
    end = time.time();

    np.save(save_root + 'short/'+ input_filename,input_raw);
    np.save(save_root + 'long/'+gt_filename,gt_image);
    save_file.write('./SID/short/'+input_filename+'.npy' + ' ' + './SID/long/'+gt_filename+'.npy' + '\n');

    print('dalong log : readdata into memory for one image consumes  {} s'.format(end - start));
save_file.close();
