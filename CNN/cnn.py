import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import warnings
warnings.filterwarnings('ignore')

class ToyNet(nn.Module):
    def __init__(self, classes=10):
        # input batchsize * 3 * 32 * 32
        # c * 2x * 2x -> conv -> 2c * x * x
        # floor((2x + 2p - k) / s) + 1 = x
        # let s = 2 where s is stride
        # k = 2p + 2 or 2p + 1
        # k is odd, so k = 2p + 1, where k is kernel size, p is padding
        # so padding 1, kernel size 3, stride 2
        super(ToyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1) # out 16 * 16
        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1) # out 8 * 8
        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1) # out 4 * 4
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1) # out 2 * 2
        self.conv5 = nn.Conv2d(512, 1024, 3, 2, 1) # out 1 * 1
        self.fc = nn.Linear(1024, 10)
        self.ConvBlock = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            self.conv2,
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            self.conv3,
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            self.conv4,
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            self.conv5,
            # nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        ) 
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        x = self.ConvBlock(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        x = self.logsoftmax(x)
        return x


