import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0,path);

caffe_path = '/usr/local/caffe/python';

demosaic_path = '/home/yuanxl/demosaicnet-master/';

add_path(caffe_path);
add_path(demosaic_path);

