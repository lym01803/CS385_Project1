import numpy as np
import torch
import pickle
import sys
import os
import random
from tqdm import tqdm

sys.path.append('..')
from utils import *

class LassoRegression:
    def __init__(self, dim):
        self.dim = dim
        self.W = torch.zeros(self.dim).cuda()
    
    def train(self, X, Y, thres_max=1.0, thres_min=1e-5):
        threshold = thres_max
        while threshold >= thres_min * 0.99:
            for d in tqdm(range(self.dim)):
                R = Y - torch.matmul(X, self.W) + X[:, d] * self.W[d]
                X_2 = torch.matmul(X[:, d], X[:, d])
                Wd = torch.matmul(R, X[:, d]) / X_2
                Wd = torch.sign(Wd) * torch.max(torch.zeros_like(Wd), torch.abs(Wd) - threshold / X_2)
                self.W[d] = Wd
            threshold *= 0.1
    
    def predict(self, X):
        return torch.matmul(X, self.W)

def Adjust(P, minnum=0, maxnum=9):
    return torch.round(torch.max(torch.ones_like(P)*minnum, torch.min(torch.ones_like(P)*maxnum, P))).long()

if __name__ == '__main__':
    # one multi-classifier
    '''
    with open('../data/data.pkl', 'rb') as f:
        D = pickle.load(f)
    X = D['train']['data']
    X = np.hstack((X.reshape(-1, X.shape[1]), np.ones((X.shape[0], 1))))
    X = torch.tensor(torch.from_numpy(X), dtype=torch.float32).cuda()
    Y = D['train']['label']
    Y = torch.tensor(torch.from_numpy(Y), dtype=torch.float32).cuda()
    LaR = LassoRegression(dim=X.shape[1])
    LaR.train(X, Y)
    # print(LaR.W)
    P = LaR.predict(X)
    # print(P)
    P = Adjust(P)
    # print(P)
    total = X.shape[0]
    correct = 0
    for num in range(10):
        correct += torch.sum(P[Y == num] == num).item()
    print('Test: correct / total = {} / {} = {}'.format(correct, total, correct / total))
    
    X = D['test']['data']
    X = np.hstack((X.reshape(-1, X.shape[1]), np.ones((X.shape[0], 1))))
    X = torch.tensor(torch.from_numpy(X), dtype=torch.float32).cuda()
    Y = D['test']['label']
    Y = torch.tensor(torch.from_numpy(Y), dtype=torch.float32).cuda()
    P = LaR.predict(X)
    P = Adjust(P)
    total = X.shape[0]
    correct = 0
    for num in range(10):
        correct += torch.sum(P[Y == num] == num).item()
    print('Test: correct / total = {} / {} = {}'.format(correct, total, correct / total))
    '''
    # 10 binary-classifier
    with open('../data/data.pkl', 'rb') as f:
        D = pickle.load(f)
    X = D['train']['data']
    X = np.hstack((X.reshape(-1, X.shape[1]), np.ones((X.shape[0], 1))))
    X = torch.tensor(torch.from_numpy(X), dtype=torch.float32).cuda()
    Y = D['train']['label']
    Y = label2onehot(Y)
    Y = torch.tensor(torch.from_numpy(Y), dtype=torch.float32).cuda()
    LaRs = [LassoRegression(dim=X.shape[1]) for i in range(10)]
    for num in tqdm(range(10)):
        LaRs[num].train(X, Y[:, num])
    total = X.shape[0]
    correct = 0
    Ps = []
    for num in range(10):
        Ps.append(LaRs[num].predict(X))
    Ps = (torch.stack(Ps).T).cuda()
    Ps = torch.argmin((Ps-1.0)**2, dim=1)
    Gr = torch.argmax(Y, dim=1)
    for num in range(10):
        correct += torch.sum(Ps[Gr == num] == num).item()
    print('Test: correct / total = {} / {} = {}'.format(correct, total, correct / total))
    
    X = D['test']['data']
    X = np.hstack((X.reshape(-1, X.shape[1]), np.ones((X.shape[0], 1))))
    X = torch.tensor(torch.from_numpy(X), dtype=torch.float32).cuda()
    Y = D['test']['label']
    Y = label2onehot(Y)
    Y = torch.tensor(torch.from_numpy(Y), dtype=torch.float32).cuda()
    
    total = X.shape[0]
    correct = 0
    Ps = []
    for num in range(10):
        Ps.append(LaRs[num].predict(X))
    Ps = (torch.stack(Ps).T).cuda()
    Ps = torch.argmin((Ps-1.0)**2, dim=1)
    Gr = torch.argmax(Y, dim=1)
    for num in range(10):
        correct += torch.sum(Ps[Gr == num] == num).item()
    print('Test: correct / total = {} / {} = {}'.format(correct, total, correct / total))
