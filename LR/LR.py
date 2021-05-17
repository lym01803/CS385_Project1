import torch
import numpy

class LogisticRegression:
    def __init__(self, dim=128):
        self.dim = dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        self.W = torch.zeros(dim).to(self.device)
        self.W.requires_grad = False # I do NOT use autograd. I calculate gradient by myself.
        torch.nn.init.normal_(self.W, mean=0.0, std=1.0)

        # For training
        self.epoch = 0
        
        # For adam
        self.v = torch.zeros(dim).to(self.device)
        self.v.requires_grad = False
        self.G = torch.tensor(0.0).to(self.device)
        self.G.requires_grad = False
    
    def train(self, X, Y, Use_adam=False, lr=0.001, gamma=0.9, beta=0.99, eps=1e-4):
        # X: (batch, dim), Y: (batch, )
        X = X.view(-1, self.dim)
        Y = Y.view(-1)
        P = torch.sigmoid(torch.matmul(X, self.W))
        g = torch.matmul(Y - P, X)
        if Use_adam:
            self.v = gamma * self.v + (1.0 - gamma) * g
            self.G = beta * self.G + (1.0 - beta) * torch.sum(g * g)
            self.v /= (1.0 - gamma)
            self.G /= (1.0 - beta)
            g = self.v / torch.sqrt(self.G + eps)
        self.W += lr * g
