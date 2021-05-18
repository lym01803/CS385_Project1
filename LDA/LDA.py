import numpy as np
import torch
import math

class LDA_model:
    def __init__(self, dim, X_pos, X_neg):
        self.dim = dim
        # num * dim
        self.X_pos = X_pos.cuda()
        self.X_neg = X_neg.cuda()
        self.mu_pos = None
        self.mu_neg = None
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

    def fit(self):
        self.n_pos = self.X_pos.shape[0] / (self.X_pos.shape[0] + self.X_neg.shape[0])
        self.n_neg = self.X_neg.shape[0] / (self.X_pos.shape[0] + self.X_neg.shape[0])
        self.mu_pos = torch.sum(self.X_pos, dim=0) / self.X_pos.shape[0]
        self.mu_neg = torch.sum(self.X_neg, dim=0) / self.X_neg.shape[0]
        self.Sigma2_pos = torch.matmul(self.X_pos.T, self.X_pos) / self.X_pos.shape[0] - torch.matmul(self.mu_pos.view(-1, 1), self.mu_pos.view(1, -1))
        self.Sigma2_neg = torch.matmul(self.X_neg.T, self.X_neg) / self.X_neg.shape[0] - torch.matmul(self.mu_neg.view(-1, 1), self.mu_neg.view(1, -1))
        self.SW = self.n_pos * self.Sigma2_pos + self.n_neg * self.Sigma2_neg
        self.beta = torch.matmul(torch.inverse(self.SW), self.mu_pos - self.mu_neg)
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
        p = - ((p - self.mu_pos) ** 2) / (2 * self.sigma2_pos) 
        return 1.0 / torch.sqrt(2 * math.pi * self.sigma2_pos) * torch.exp(p)