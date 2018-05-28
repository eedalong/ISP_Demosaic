import numpy as np


'''
This is an implementation for extract 4 channels from a raw data

receive a raw data as 3D array of size HxWx1

only support GRBG and RGGB

'''
def create_CFA(raw,bayer_type = 'GRBG'):
    CFA = np.zeros((int(raw.shape[1]/2),int(raw.shape[2]/2),4));
    if bayer_type == 'GRBG':

        CFA[:,:,0] = raw[::2,::2,0];  # G
        CFA[:,:,1] = raw[::2,1::2,0]; # R
        CFA[:,:,2] = raw[1::2,::2,0]; # B
        CFA[:,:,3] = raw[1::2,1::2,0];# G
    elif bayer_type == 'RGGB':
        CFA[:,:,0] = raw[0,::2,1::2]; # G
        CFA[:,:,1] = raw[0,::2,::2];  # R
        CFA[:,:,2] = raw[0,1::2,1::2];# B
        CFA[:,:,3] = raw[0,1::2,::2]; # G
    else :
        print('Unsupported bayer type, EXIT!');
        exit();
    return CFA.astype('float');


