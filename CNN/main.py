from torch.utils.data import dataloader
from cnn import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import torchvision
import torchvision.utils as  vutils
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import pickle
from tqdm import tqdm 

from PIL import Image

import sys, os

class ImgDataSet(Dataset):
    def __init__(self):
        super(ImgDataSet, self).__init__()
        self.img = list()
        self.label = list()
    
    def add(self, x, y):
        self.img.append(x)
        self.label.append(y)
    
    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]

    def __len__(self):
        return len(self.img)

def makedataset(path):
    labeldict = dict()
    with open(os.path.join(path, 'label.txt'), 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            li = line.strip().split()
            if len(li) > 1:
                integer = int(li[1])
                if integer > 9:
                    integer -= 10
                labeldict[li[0]] = int(integer)
    imgSet = ImgDataSet()
    for r, d, files in os.walk(path):
        for file in tqdm(files):
            if file.strip().split('.')[-1] == 'png':
                img = Image.open(os.path.join(path, file))
                img = converter(img)
                img = normalizer(img)
                label = labeldict[file]
                imgSet.add(img, label)
    return DataLoader(dataset=imgSet, batch_size=64, shuffle=True)

trainpath = '../data/processed/train'
validpath = '../data/processed/test'
converter = transforms.ToTensor()
normalizer = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

trainloader = makedataset(trainpath)
validoader = makedataset(validpath)

model = ToyNet().cuda()
opt = optim.Adam(model.parameters(), lr=1e-3)
lossfunc = nn.NLLLoss()
scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)
for epoch in tqdm(range(100)):
    model.train()
    loss_ = []
    for i, (data, label) in tqdm(enumerate(trainloader)):
        data = data.cuda()
        label = label.cuda()
        model.zero_grad()
        out = model(data)
        loss = lossfunc(out, label)
        loss_.append(loss.item())
        loss.backward()
        opt.step()
    scheduler.step()
    print('Averaged loss: ', sum(loss_) / len(loss_))
    if (epoch + 1) % 10 == 0:
        # Valid
        total = 0
        correct = 0
        model.eval()
        for i, (data, label) in tqdm(enumerate(validoader)):
            data = data.cuda()
            label = label.cuda()
            with torch.no_grad():
                out = model(data)
                res = torch.argmax(out, dim=-1)
                total += res.shape[0]
                correct += torch.sum(res == label).item()
        print('Acc = correct / total = {} / {} = {}'.format(correct, total, correct / total))


