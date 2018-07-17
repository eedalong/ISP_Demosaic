import torch
import matplotlib.pyplot as plt
import os
def save_checkpoint(model,path_checkpoint):

    torch.save(model.state_dict(),path_checkpoint);

class AverageMeter:
    def __init__(self):
        self.n = 0;
        self.value = 0;
    def update(self,value,count = 1):
        self.value = (self.value * self.n + value * count) / (self.n + count);
        self.n = self.n + count;
        return ;


class Recorder:
    def __init__(self,save_freq = 1,save_dir = './'):
        self.records = {};
        self.freq = save_freq;
        self.save_dir = save_dir;
    def add_record(self,record_name,value,steps):
        if self.records.get(record_name,'dalong') == 'dalong':
           self.records[record_name] = {};
        self.records[record_name][steps] = value;
        if steps % self.freq == 0:
            pass
            #self.DrawGraph(record_name);
    def DrawGraph(self,record_name):
        figure = plt.figure(record_name);
        plt.plot(self.records[record_name].keys(),self.records[record_name].values());
        figure.savefig(os.path.join(self.save_dir,record_name+'.png'));

def save_logs(root_path,args):
    log_file = open(os.path.join(root_path,'log_file'),'w');
    log = vars(args);
    for key in log :
        log_file.write(key + ' : ' + str(log[key]) + '\n');
    log_file.close();
def RandomCrop(size,raw,data):
    h,w = raw.shape[0],raw.shape[1];
    th,tw = size;
    x1 = random.randint(0,w -tw);
    y1 = random.randint(0,h - th);
    return raw[y1:y1+th,x1:x1+tw,:],data[y1*2:y1*2+th*2,x1*2:x1*2+tw*2,:];
def RandomFLipH(raw,data):
	if random.random()< 0.5:
		return np.flip(raw,axis = 1).copy(),np.flip(data,axis = 1).copy();
	return raw,data;
def RandomFlipV(raw,data):
 	if random.random()< 0.5:
		return np.flip(raw,axis = 0).copy(),np.flip(data,axis = 0).copy();
	return raw,data;
def RandomTranspose(raw,data):
    if random.random()< 0.5:
        return np.transpose(raw,(1,0,2)).copy(),np.transpose(data,(1,0,2)).copy();
    return raw,data;

def collate_fn(batch):
    raw = batch[0][0];
    data = batch[0][1]
    for index in range(1,len(batch)):
        raw = torch.cat((raw,batch[index][0]),0);
        data = torch.cat((data,batch[index][1]),0);
    return raw,data;


def main():
	print('This is for saving models');

if __name__ == '__main__':
	main();

