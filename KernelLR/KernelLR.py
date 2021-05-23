import torch
import numpy as np
import sys
import pickle
import random
from torch.autograd import backward
from tqdm import tqdm

sys.path.append('..')
from utils import *

class KernelLR:
    def __init__(self, X, Y, batch_size=256, w=10.0):
        self.dim = X.shape[1]
        self.X = X
        self.Y = Y
        self.K = None
        self.Alpha = torch.zeros(self.X.shape[0]).cuda()
        self.Alpha.requires_grad = False
        self.batch_size = batch_size
        self.w = w

        self.v = torch.zeros(self.X.shape[0]).cuda()
        self.v.requires_grad = False
        self.G = torch.tensor(0.0).cuda()
        self.G.requires_grad = False


    def fit(self, max_epoch=500, Use_Adam=True, lr=1.0, gamma=0.9, beta=0.999, eps=1e-4):
        X = self.X
        Y = self.Y
        n = X.shape[0]
        self.K = torch.zeros((n, n)).cuda()
        batch_num = (n - 1) // self.batch_size + 1
        print('calculate K matrix')
        idx = [i for i in range(n)]
        for bi in tqdm(range(batch_num)):
            biidx = idx[bi*self.batch_size: bi*self.batch_size+self.batch_size]
            biidx = torch.tensor(biidx, dtype=torch.long).cuda()
            Xbi = torch.index_select(X, 0, biidx)
            for bj in range(batch_num):
                bjidx = idx[bj*self.batch_size: bj*self.batch_size+self.batch_size]
                bjidx = torch.tensor(bjidx, dtype=torch.long).cuda()
                Xbj = torch.index_select(X, 0, bjidx)
                X2i = torch.sum(Xbi * Xbi, dim=1)
                X2j = torch.sum(Xbj * Xbj, dim=1)
                Xij = torch.matmul(Xbi, Xbj.T)
                Dij = X2j - 2.0 * Xij + X2i.view(-1, 1)
                Dij = torch.exp(- Dij / self.w)
                idxi, idxj = torch.meshgrid(biidx, bjidx)
                self.K[idxi, idxj] = Dij
        for epoch in tqdm(range(max_epoch)):
            random.shuffle(idx)
            for b in range(batch_num):
                batch_idx = idx[b*self.batch_size: b*self.batch_size+self.batch_size]
                batch_idx = torch.tensor(batch_idx, dtype=torch.long).cuda()
                # Xb = torch.index_select(X, 0, batch_idx)
                Yb = torch.index_select(Y, 0, batch_idx)
                Kb = torch.index_select(self.K, 0, batch_idx)
                Pb = torch.matmul(Kb, self.Alpha)
                Pb = torch.sigmoid(Pb)
                g = torch.matmul(Kb.T, Yb - Pb)
                if Use_Adam:
                    self.v = gamma * self.v + (1.0 - gamma) * g
                    self.G = beta * self.G + (1.0 - beta) * torch.sum(g * g)
                    g = (self.v / (1.0 - gamma)) / torch.sqrt(self.G / (1.0 - beta) + eps)
                self.Alpha += lr * g
            if epoch % 100 == 0:
                lr *= 0.1
    
    def predict(self, Xt):
        X = self.X
        n = X.shape[0]
        nt = Xt.shape[0]
        idx = [i for i in range(n)]
        idxt = [i for i in range(nt)]
        batch_size = self.batch_size
        batch_num = (n - 1) // batch_size + 1
        batch_numt = (nt - 1) // batch_size + 1
        TempK = torch.zeros((nt, n)).cuda()
        for bt in tqdm(range(batch_numt)):
            btidx = idxt[bt*batch_size: bt*batch_size+batch_size]
            btidx = torch.tensor(btidx, dtype=torch.long).cuda()
            Xbt = torch.index_select(Xt, 0, btidx)
            for b in range(batch_num):
                bidx = idx[b*batch_size: b*batch_size+batch_size]
                bidx = torch.tensor(bidx, dtype=torch.long).cuda()
                Xb = torch.index_select(X, 0, bidx)
                X2t = torch.sum(Xbt * Xbt, dim=1)
                X2 = torch.sum(Xb * Xb, dim=1)
                Xij = torch.matmul(Xbt, Xb.T)
                Dij = X2 - 2.0 * Xij + X2t.view(-1, 1)
                Dij = torch.exp(- Dij / self.w)
                idxbt, idxb = torch.meshgrid(btidx, bidx)
                TempK[idxbt, idxb] = Dij
        P = torch.matmul(TempK, self.Alpha)
        P = torch.sigmoid(P)
        return P
        
    def rbf(self, x, y):
        d = x - y 
        return torch.exp(- torch.sum(d * d) / self.w)

if __name__ == '__main__':
    with open('../data/data.pkl', 'rb') as f:
        D = pickle.load(f)
    sample = 10000
    X = D['train']['data'][:sample]
    '''
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 2], 
        [2, 3],
        [3, 2],
        [3, 3],
        [-5, 5],
        [-6, 5],
        [-5, 4],
        [-4, 6],
        [4, -2], 
        [4, -1],
        [3, -4],
        [3, -4],
        [-1, -1],
        [-2, -2],
        [-3, -3],
        [-2, -3]
    ])
    '''
    X = np.hstack((X.reshape(-1, X.shape[1]), np.ones((X.shape[0], 1))))
    X = torch.tensor(torch.from_numpy(X), dtype=torch.float32).cuda()
    Y = D['train']['label'][:sample]
    # Y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
    Y = label2onehot(Y)
    Y = torch.tensor(torch.from_numpy(Y), dtype=torch.float32).cuda()
    models = []
    for num in tqdm(range(10)):
        models.append(KernelLR(X, Y[:, num]))
        models[num].fit()
    
    X = D['test']['data'][:sample]
    '''
    X = np.array([
        [0.2, -0.1],
        [0.2, 0.2],
        [3, 5],
        [4, 7],
        [-3.2, 2.9],
        [-2.0, 4.1],
        [5.0, -2.0],
        [3.0, -5.0],
        [-2.1, -2.4],
        [-3.5, -3.7]
    ])
    '''
    X = np.hstack((X.reshape(-1, X.shape[1]), np.ones((X.shape[0], 1))))
    X = torch.tensor(torch.from_numpy(X), dtype=torch.float32).cuda()
    Y = D['test']['label'][:sample]
    # Y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    Y = label2onehot(Y)
    Y = torch.tensor(torch.from_numpy(Y), dtype=torch.float32).cuda()
    
    Ps = []
    for num in range(10):
        Ps.append(models[num].predict(X))
        # print(models[num].Alpha)
    Ps = (torch.stack(Ps).T).cuda()
    # print(Ps.tolist())
    predict_label = torch.argmax(Ps, dim=1)
    # print(predict_label.tolist())
    ground_truth = torch.argmax(Y, dim=1)
    total = [0 for i in range(11)]
    correct = [0 for i in range(11)]
    for num in range(10):
        total[num] = torch.sum(ground_truth == num).item()
        correct[num] = torch.sum(predict_label[ground_truth == num] == num).item()
        total[-1] += total[num]
        correct[-1] += correct[num]
        print('Number : {} : correct / total = {} / {} = {}'.format(num, correct[num], total[num], correct[num] / total[num]))
    print('Test Total ----- correct / total = {} / {} = {}'.format(correct[-1], total[-1], correct[-1] / total[-1]))
