import numpy as np
import torch
import math
import sys
import pickle
from tqdm import tqdm

sys.path.append('..')
from utils import *

class LDA_model:
    def __init__(self, dim, X_pos, X_neg):
        self.dim = dim
        # num * dim
        self.X_pos = X_pos.cuda()
        self.X_neg = X_neg.cuda()
        self.mu_pos = None
        self.mu_neg = None
        self.mu_pos_beta = None
        self.mu_neg_beta = None
        self.Sigma2_pos = None
        self.Sigma2_neg = None
        self.n_pos = None
        self.n_neg = None
        self.SW = None
        self.beta = None
        self.sigma2_pos = None
        self.sigma2_neg = None
        self.threshold = None
        self.pos_indicator = None

    def fit(self, eps=1e-4):
        self.n_pos = self.X_pos.shape[0] / (self.X_pos.shape[0] + self.X_neg.shape[0])
        self.n_neg = self.X_neg.shape[0] / (self.X_pos.shape[0] + self.X_neg.shape[0])
        self.mu_pos = torch.sum(self.X_pos, dim=0) / self.X_pos.shape[0]
        self.mu_neg = torch.sum(self.X_neg, dim=0) / self.X_neg.shape[0]
        self.Sigma2_pos = torch.matmul(self.X_pos.T, self.X_pos) / self.X_pos.shape[0] - torch.matmul(self.mu_pos.view(-1, 1), self.mu_pos.view(1, -1))
        self.Sigma2_neg = torch.matmul(self.X_neg.T, self.X_neg) / self.X_neg.shape[0] - torch.matmul(self.mu_neg.view(-1, 1), self.mu_neg.view(1, -1))
        self.SW = self.n_pos * self.Sigma2_pos + self.n_neg * self.Sigma2_neg
        self.beta = torch.matmul(torch.inverse(self.SW + eps*torch.eye(self.dim).cuda()), self.mu_pos - self.mu_neg)
        self.mu_pos_beta = torch.matmul(self.mu_pos, self.beta)
        self.mu_neg_beta = torch.matmul(self.mu_neg, self.beta)
        self.sigma2_pos = torch.matmul(torch.matmul(self.beta, self.Sigma2_pos), self.beta)
        self.sigma2_neg = torch.matmul(torch.matmul(self.beta, self.Sigma2_neg), self.beta)
        self.threshold = torch.matmul(self.mu_neg, self.beta) * torch.sqrt(self.sigma2_pos) + torch.matmul(self.mu_pos, self.beta) * torch.sqrt(self.sigma2_neg)
        self.threshold /= torch.sqrt(self.sigma2_pos) + torch.sqrt(self.sigma2_neg)
        self.pos_indicator = torch.matmul(self.mu_pos, self.beta) - self.threshold

    def proj(self, X):
        if self.beta is None:
            return None
        return torch.matmul(X.cuda(), self.beta)
    
    def predict(self, X):
        if self.beta is None:
            return None
        p = self.proj(X)
        # print(p.shape)
        p = - ((p - self.mu_pos_beta) ** 2) / (2 * self.sigma2_pos) 
        return 1.0 / torch.sqrt(2 * math.pi * self.sigma2_pos) * torch.exp(p)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = '../data/data.pkl'
    print('loading...')
    with open(filepath, 'rb') as f:
        D = pickle.load(f)
    print('loaded')
    X = D['train']['data']
    X = X.reshape(-1, X.shape[1])
    X = torch.tensor(torch.from_numpy(X), dtype=torch.float32).cuda()
    Y = D['train']['label']
    Y = torch.tensor(torch.from_numpy(Y), dtype=torch.float32).cuda()

    LDAs = []
    for i in tqdm(range(10)):
        LDAs.append(LDA_model(
            dim=X.shape[1],
            X_pos=X[Y==i],
            X_neg=X[Y!=i]
        ))
        LDAs[-1].fit()
    
    X = D['test']['data']
    X = X.reshape(-1, X.shape[1])
    X = torch.tensor(torch.from_numpy(X), dtype=torch.float32).cuda()
    Y = D['test']['label']
    Y = label2onehot(Y)
    Y = torch.tensor(torch.from_numpy(Y), dtype=torch.float32).cuda()

    idx = [i for i in range(X.shape[0])]
    batch_size = 256
    batch_num = (X.shape[0] - 1) // batch_size + 1
    total = [0 for i in range(11)]
    correct = [0 for i in range(11)]
    for b in tqdm(range(batch_num)):
        batch_idx = idx[b*batch_size: b*batch_size+batch_size]
        batch_idx = torch.tensor(batch_idx, dtype=torch.long).cuda()
        Xb = torch.index_select(X, 0, batch_idx)
        Yb = torch.index_select(Y, 0, batch_idx)
        Ps = []
        for model_num in range(10):
            Ps.append(LDAs[model_num].predict(Xb))
        Pb = (torch.stack(Ps).T).cuda()
        predict_label = torch.argmax(Pb, dim=1)
        ground_truth = torch.argmax(Yb, dim=1)
        # print(predict_label.shape, ground_truth.shape)
        total[-1] += predict_label.shape[0]
        correct[-1] += torch.sum(predict_label == ground_truth).item()
        for number in range(10):
            total[number] += ground_truth[ground_truth == number].shape[0]
            correct[number] += torch.sum(predict_label[ground_truth == number] == number).item()
    for number in range(10):
        print('Number : {} : correct / total = {} / {} = {}'.format(number, correct[number], total[number], correct[number] / total[number]))
    print('Test Total ----- correct / total = {} / {} = {}'.format(correct[-1], total[-1], correct[-1] / total[-1]))
