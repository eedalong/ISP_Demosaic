'''
This is an implementation of Noise estimation from

KPN and from Practical Poissonian-Gaaussian noise modeling and fitting for single-image raw-data

sigma paramters are readed from DNG meta file

'''

import numpy as np

'''
Params:
	img: numpy array
    sigma_p : possion noise
	sigma_g: read noise
    two sigma parameters can be read from DNG file
'''
def NoiseEstimation(img,black_level,sigma_p,sigma_g):
    img = img - black_level;
    sigma = np.sqrt(sigma_p + sigma_p * max(img,0));
    return sigma;


