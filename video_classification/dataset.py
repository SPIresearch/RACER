import os
import torch
import numpy as np
from torch.nn import functional as F

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, file_path, trans):
        self.data = []
        self.labels = []
        self.file_path = file_path
        self.N_CLASSES = 50
        self.data_dir = data_dir
        self.trans = trans
        self._read_file(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.file_path, self.data[idx])
        data = torch.from_numpy(np.load(path)['data'])
        
        label = self.labels[idx]
        data = self.trans(data)
        return data.unsqueeze(0), label, self.data[idx]

    def _read_file(self, file_path):
        with open(file_path) as f:
            for line in f:
                pid, path, label = line.strip().split(' ')
                self.data.append( os.path.join(self.data_dir, pid+'_mouth', path+'.npz'))
                self.labels.append(int(label))

# dataset = RadarDataset([0,1,2], '/home/spi/xuyinsong/lip/')
# for i in range(150):
#     data, label = dataset.__getitem__(i)
#     print(data.shape)


