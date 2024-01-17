import torch
import torch.nn as nn
import math
import numpy as np
from .resnet import ResNet, BasicBlock
from torchvision.models import resnet18
from backbone.cnn import  Identity
# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)

class R18(nn.Module):
    def __init__( self):
        super(R18, self).__init__()
        relu_type = 'prelu'
        self.frontend_nout = 64
        self.backend_out = 512
        frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
        self.frontend3D = nn.Sequential(
                    nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                    nn.BatchNorm3d(self.frontend_nout),
                    frontend_relu,
                    nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))

        self.trunk = resnet18(pretrained=True)

         

    def forward(self, x):
        B, C, T, H, W = x.size()

        x = self.frontend3D(x)
        Tnew = x.shape[2]    # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor( x )
        x = self.trunk.layer1(x)
        x = self.trunk.layer2(x)
        x = self.trunk.layer3(x)
        x = self.trunk.layer4(x)
        x = self.trunk.avgpool(x)
        x = x.view(x.size(0), -1)
        x = x.view(B, Tnew, x.size(1))
        return x.mean(dim=1)
