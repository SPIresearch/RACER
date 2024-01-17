import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T

class Freq_mask(nn.Module):
    def __init__(self, h=20):
        super(Freq_mask, self).__init__()
        self.h = h

    def forward(self, x):
        # x(1,freq,time)
        h_max = x.size(1) #torch.randint(0, h - th + 1, size=(1, )).item()
        f = torch.randint(low=0, high=self.h, size=(1, )).item()
        f0 = torch.randint(0, h_max-f, size=(1, )).item()
        x[:,f0:f0+f,:] = 0
        return x

class Time_mask(nn.Module):
    def __init__(self, w=50):
        super(Time_mask, self).__init__()
        self.w = w

    def forward(self, x):
        # x(1, freq,time)
        w_max = x.size(2)
        t = torch.randint(low=0, high=self.w, size=(1, )).item()
        t0 = torch.randint(0, w_max-t, size=(1, )).item()
        x[:,:,t0:t0+t] = 0
        return x
# for the left, top, right and bottom borders respectively.

class Freq_flip(nn.Module):
    def __init__(self):
        super(Freq_flip, self).__init__()

    def forward(self, x):
        if torch.rand(1) < 0.5:
            return F.vflip(x)
        return x


class Time_pad(nn.Module):
    def __init__(self, w=500):
        super(Time_pad, self).__init__()
        self.w = w
    
    def forward(self, x):
        w = x.size(2)
        pad_time = torch.randint(low=0, high=self.w, size=(1, )).item()
        if torch.rand(1) < 0.5:
            F.pad(x,(pad_time,0,0,0), 0, "constant")
            x = x[:,:,:w]
        else:
            F.pad(x,(0,0,pad_time,0), 0, "constant")
            x = x[:,:,-w:]
        return x

def get_transforms(args):
    trans = []
    if not args.no_freq_mask:
        trans.append(Freq_mask())
    if not args.no_time_mask:
        trans.append(Time_mask())
    if not args.no_freq_flip:
        trans.append(Freq_flip())
    if not args.no_time_pad:
        trans.append(Time_pad())
    trans.append(T.Resize((400,1500)))
    train_trans = T.Compose(trans)
    val_trans = T.Resize((400,1500))
    return train_trans, val_trans