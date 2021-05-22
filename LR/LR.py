import torch
import numpy as np
import sys
import pickle
import random

sys.path.append('..')
from utils import *

class LogisticRegression:
    def __init__(self, dim=128):
        self.dim = dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        self.W = torch.zeros(dim).to(self.device)
        self.W.requires_grad = False # I do NOT use autograd. I calculate gradient by myself.
        # torch.nn.init.normal_(self.W, mean=0.0, std=1.0)

        # For training
        self.epoch = 0
        
        # For adam
        self.v = torch.zeros(dim).to(self.device)
        self.v.requires_grad = False
        self.G = torch.tensor(0.0).to(self.device)
        self.G.requires_grad = False
    
    def train(self, X, Y, Use_adam=False, lr=0.001, gamma=0.9, beta=0.999, eps=1e-4, ridge_lambda=None, lasso_lambda=None):
        # X: (batch, dim), Y: (batch, )
        X = X.view(-1, self.dim)
        Y = Y.view(-1)
        P = torch.sigmoid(torch.matmul(X, self.W))
        g = torch.matmul(Y - P, X)
        # objective: max L' = L - lambda * norm2(W)
        # gradient' = gradient - 2 * lambda * W
        if not ridge_lambda is None:
            g -= 2.0 * ridge_lambda * self.W
        elif not lasso_lambda is None:
            g -= lasso_lambda * torch.sign(self.W)
        if Use_adam:
            self.v = gamma * self.v + (1.0 - gamma) * g
            self.G = beta * self.G + (1.0 - beta) * torch.sum(g * g)
            # self.v /= (1.0 - gamma)
            # self.G /= (1.0 - beta)
            g = (self.v / (1.0 - gamma)) / torch.sqrt(self.G / (1.0 - beta) + eps)
        self.W += lr * g
    
    def predict(self, X):
        X = X.view(-1, self.dim)
        P = torch.sigmoid(torch.matmul(X, self.W))
        return P

    def proj(self, X):
        return torch.matmul(X.view(-1, self.dim), self.W)

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
    X = np.hstack((X.reshape(-1, X.shape[1]), np.ones((X.shape[0], 1))))
    X = torch.tensor(torch.from_numpy(X), dtype=torch.float32).cuda()
    Y = D['train']['label']
    Y = label2onehot(Y)
    Y = torch.tensor(torch.from_numpy(Y), dtype=torch.float32).cuda()
    models = [LogisticRegression(dim=X.shape[1]) for i in range(10)]
    idx = [i for i in range(X.shape[0])]

    lr = 0.1
    for epoch in tqdm(range(100)):
        random.shuffle(idx)
        batch_size = 256
        batch_num = (X.shape[0] - 1) // batch_size + 1
        for b in tqdm(range(batch_num)):
            batch_idx = idx[b*batch_size: b*batch_size+batch_size]
            batch_idx = torch.tensor(batch_idx, dtype=torch.long).cuda()
            Xb = torch.index_select(X, 0, batch_idx)
            Yb = torch.index_select(Y, 0, batch_idx)
            for model_num in range(10):
                models[model_num].train(Xb, Yb[:, model_num], lr=lr, Use_adam=True)
        if (epoch + 1) % 10 == 0:
            total = [0 for i in range(11)]
            correct = [0 for i in range(11)]
            for b in tqdm(range(batch_num)):
                batch_idx = idx[b*batch_size: b*batch_size+batch_size]
                batch_idx = torch.tensor(batch_idx, dtype=torch.long).cuda()
                Xb = torch.index_select(X, 0, batch_idx)
                Yb = torch.index_select(Y, 0, batch_idx)
                Ps = []
                for model_num in range(10):
                    Ps.append(models[model_num].predict(Xb))
                Pb = (torch.stack(Ps).T).cuda()
                predict_label = torch.argmax(Pb, dim=1)
                ground_truth = torch.argmax(Yb, dim=1)
                # print(predict_label.shape)
                total[-1] += predict_label.shape[0]
                correct[-1] += torch.sum(predict_label == ground_truth).item()
                for number in range(10):
                    total[number] += ground_truth[ground_truth == number].shape[0]
                    correct[number] += torch.sum(predict_label[ground_truth == number] == number).item()
            for number in range(10):
                print('Number : {} : correct / total = {} / {} = {}'.format(number, correct[number], total[number], correct[number] / total[number]))
            print('Total ----- correct / total = {} / {} = {}'.format(correct[-1], total[-1], correct[-1] / total[-1]))
        # if (epoch + 1) % 25 == 0:
            # lr /= 2
    # Is W sparse ?
    for number in range(10):
        w = models[number].W
        print("number : {} ; norm2(W): {}\nnorm1(W): {}\nnorm1/norm2: {}".format(number, torch.norm(w).item(), torch.norm(w, 1).item(), torch.norm(w, 1).item() / torch.norm(w, 2).item()))
    # For plot
    proj_for_plot = [[[], []] for num in range(10)]
    print('making data for plot')
    batch_size = 256
    batch_num = (X.shape[0] - 1) // batch_size + 1
    for b in tqdm(range(batch_num)):
        batch_idx = idx[b*batch_size: b*batch_size+batch_size]
        batch_idx = torch.tensor(batch_idx, dtype=torch.long).cuda()
        Xb = torch.index_select(X, 0, batch_idx)
        Yb = torch.index_select(Y, 0, batch_idx)
        for model_num in range(10):
            # models[model_num].train(Xb, Yb[:, model_num], lr=lr, Use_adam=True)
            # print(Yb.shape, (Yb[:, model_num] == 1).shape, Xb[Yb[:, model_num] == 1].shape)
            Xb_pos_proj = models[model_num].proj(Xb[Yb[:, model_num] == 1]).tolist()
            Xb_neg_proj = models[model_num].proj(Xb[Yb[:, model_num] == 0]).tolist()
            proj_for_plot[model_num][0] += Xb_pos_proj
            proj_for_plot[model_num][1] += Xb_neg_proj
    with open('./proj_for_plot.pkl', 'wb') as f:
        pickle.dump(proj_for_plot, f)

    # For test
    X = D['test']['data']
    X = np.hstack((X.reshape(-1, X.shape[1]), np.ones((X.shape[0], 1))))
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
            Ps.append(models[model_num].predict(Xb))
        Pb = (torch.stack(Ps).T).cuda()
        predict_label = torch.argmax(Pb, dim=1)
        ground_truth = torch.argmax(Yb, dim=1)
        # print(predict_label.shape)
        total[-1] += predict_label.shape[0]
        correct[-1] += torch.sum(predict_label == ground_truth).item()
        for number in range(10):
            total[number] += ground_truth[ground_truth == number].shape[0]
            correct[number] += torch.sum(predict_label[ground_truth == number] == number).item()
    for number in range(10):
        print('Number : {} : correct / total = {} / {} = {}'.format(number, correct[number], total[number], correct[number] / total[number]))
    print('Test Total ----- correct / total = {} / {} = {}'.format(correct[-1], total[-1], correct[-1] / total[-1]))
    
    '''
    X, Y = load_data('../data/processed/extra')
    ratio = 0.8
    bord = int(X.shape[0] * ratio)
    X_train = X[ : bord]
    X_test = X[bord : ]
    Y_train = Y[ : bord]
    Y_test = Y[bord : ]
    D = {
        'train': {
            'data': X_train,
            'label': Y_train
        },
        'test': {
            'data': X_test,
            'label': Y_test
        }
    }
    with open('./extra_data.pkl', 'wb') as f:
        pickle.dump(D, f)
    '''
