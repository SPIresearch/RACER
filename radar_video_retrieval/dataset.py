import os
import torch
import numpy as np


def radar_norm(data):
    min, max = data.min(), data.max()
    data = (data - min)/(max-min)
    return data

class RVDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, file_path, radar_trans, video_trans):
        self.r_data = []
        self.v_data = []
        self.labels = []
        self.file_path = file_path
        self.N_CLASSES = 50
        self.data_dir = data_dir
        self.radar_trans = radar_trans
        self.video_trans = video_trans
        self._read_file(file_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        radar_path = os.path.join(self.file_path, self.r_data[idx])
        video_path = os.path.join(self.file_path, self.v_data[idx])

        radar = torch.from_numpy(np.load(radar_path)).float()
        video = torch.from_numpy(np.load(video_path)['data'])
        radar = radar_norm(radar)

        radar = self.radar_trans(radar.unsqueeze(0))
        video = self.video_trans(video)
        return radar.repeat(3,1,1), video.unsqueeze(0), label, self.r_data[idx] 

    def _read_file(self, file_path):
        with open(file_path) as f:
            for line in f:
                pid, path, label = line.strip().split(' ')
                self.r_data.append( os.path.join(self.data_dir, pid+'_doppler_npy', path+'.npy'))
                self.v_data.append( os.path.join(self.data_dir, pid+'_mouth', path+'.npz'))
                self.labels.append(int(label))
