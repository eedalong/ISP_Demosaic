import numpy as np
import math
import create_CFA
# now we dont apply the finer rules
MO = 1 / 0.83893;
AO_CA_K = 10.2655 - 13.5570;
alpha_R = 1.8523;
alpha_G = 0.6891;
alpha_B = 1;
# directional homogeneity measures
# default 5x5 size
def dhm(block):
    direct1 = [[2,2,2,2,2],[0,1,2,3,4]];
    direct2 = [[0,1,2,3,4],[2,2,2,2,2]];
    direct3 = [[0,1,2,3,4],[0,1,2,3,4]];
    direct4 = [[0,1,2,3,4],[4,3,2,1,0]];
    direct5 = [[2,2,2,3,4],[4,3,2,2,2]];
    direct6 = [[2,2,2,3,4],[0,1,2,2,2]];
    direct7 = [[0,1,2,2,2],[2,2,2,1,0]];
    direct8 = [[0,1,2,2,2],[2,2,2,3,4]];
    HP = np.array([-1,-1,4,-1,-1]).astype('float32');
    directs = [direct1,direct2,direct3,direct4,direct5,direct6,direct7,direct8];
    measures = 0.0;
    for direct in directs:
        measures = measures + abs(HP.dot(block[direct[0],direct[1]]));
    return measures,np.var(block);

# here we receive a CFA numpy array of 4 channels G1 R B G2
def NoiseEstimation(CFA,Tao = 3):

    minst = np.array([1000,1000,1000,1000,1000,1000]).astype('float32');
    medium = np.array([1000,1000,1000,1000,1000,1000]).astype('float32');
    maximum = np.array([1000,1000,1000,1000,1000,1000]).astype('float32');

    four = np.array([1000,1000,1000,1000,1000,1000]).astype('float32');
    five = np.array([1000,1000,1000,1000,1000,1000]).astype('float32');


    variance = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]).astype('float32');
    min_value = 1e-3;
    for channel in range(CFA.shape[-1]):
        for row in range(0,CFA.shape[0],Tao):
            for col in range(0,CFA.shape[1],Tao):
                # jude id valid
                if row - 2 < 0 or row + 2 >= CFA.shape[0] or col -2 < 0 or col + 2 >= CFA.shape[1]:
                    continue
                block = [[],[]];
                for i in range(row-2,row+2+1):
                    for j in range(col-2,col+2+1):
                         block[0].append(i);
                         block[1].append(j);

                measure,var = dhm(CFA[block[0],block[1],channel].reshape(5,5));
                # matain the first three minimum value
                if measure < minst[channel]:
                    save_measure = minst[channel];
                    save_var = variance[channel][0];
                    minst[channel] = measure;
                    variance[channel][0] = var;
                    if save_measure == 1000 :
                        continue;
                    var = save_var;
                    measure = save_measure;
                if measure < medium[channel]:
                    save_measure = medium[channel];
                    save_var = variance[channel][1];
                    medium[channel] = measure;
                    variance[channel][1] = var;
                    var = save_var;
                    measure = save_measure;
                if measure < maximum[channel]:
                    save_measure = maximum[channel];
                    save_var = variance[channel][2];

                    maximum[channel] = measure;
                    variance[channel][2] = var;

                    var = save_var;
                    measure = save_measure;
                if measure < four[channel]:
                    save_measure = four[channel];
                    save_var = variance[channel][3];

                    four[channel] = measure;
                    variance[channel][3] = var;

                    var = save_var;
                    measure = save_measure;
                if measure < five[channel]:
                    save_measure = five[channel];
                    save_var = variance[channel][4];

                    five[channel] = measure;
                    variance[channel][4] = var;

                    var = save_var;
                    measure = save_measure;

    max_value = np.max(variance,1);
    min_value = np.min(variance,1);
    sum_value = np.sum(variance,1);

    sigma = (sum_value - max_value - min_value) / 3.0;
    # estimate alpha value
    alpha_r = (sigma[1] / (sigma[2] + 1e-5));
    alpha_g = ((sigma[0] + sigma[3])/2/(sigma[2] + 1e-5));
    alpha_b = 1

    alpha = [alpha_g,alpha_r,alpha_b,alpha_g];
    alpha = np.sqrt(alpha);
    sigma_e = np.median(np.sqrt(sigma)/(alpha+1e-5));
    sigmaA = sigma_e ;
    return sigmaA;

def UNIT_TEST(sigma,file_name,step):
    from PIL import Image
    path = 'data/1.bmp';
    import os
    if not os.path.exists(file_name):
        NE = open(file_name,'w');
        NE.close();
    NE = open(file_name,'a');
    image = Image.open(path);
    w,h = image.size;
    a = np.random.uniform(0,255,size = (h,w,1));
    a[:,:,0] = np.asarray(image).astype('float32');
    noise = np.random.normal(0,np.sqrt(sigma),(h,w,1));
    a = a + noise;
    CFA = create_CFA.create_CFA(a);
    noise_level  = NoiseEstimation(CFA,step);
    NE.write(str(np.sqrt(sigma))+ ' '+ str(np.sqrt(noise_level)) + '\n');
    NE.close();

if __name__  == '__main__':
    for sigma in range(0,20):
        UNIT_TEST(sigma,'tmp',3);

