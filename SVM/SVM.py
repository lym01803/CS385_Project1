import numpy as np
import pickle
import sys
import os
from sklearn.svm import SVC
from tqdm import tqdm
import random

sys.path.append('..')
from utils import *

if __name__ == '__main__':

    print('loading...')
    with open('../data/data.pkl', 'rb') as f:
        D = pickle.load(f)
    print('loaded')
    X = D['train']['data']
    X = X.reshape(-1, X.shape[1])
    Y = D['train']['label']
    Y = label2onehot(Y)
    models = [SVC(C=1.0, kernel='rbf', cache_size=3000) for i in range(10)]

    idx = [i for i in range(X.shape[0])]
    random.shuffle(idx)
    idx = np.array(idx[:10000])
    X = X[idx]
    Y = Y[idx]

    for number in tqdm(range(10)):
        models[number].fit(X, Y[:, number])
    
    X = D['test']['data']
    X = X.reshape(-1, X.shape[1])
    Y = D['test']['label']
    P = []
    for number in tqdm(range(10)):
        p = models[number].predict(X)
        P.append(p + 0.01 * np.random.random(p.shape[0]))
    P = np.stack(P).T
    Predict = np.argmax(P, axis=1)
    total = 0
    correct = 0
    for number in range(10):
        t = np.sum(Y == number)
        c = np.sum(Predict[Y == number] == number)
        print('Number : {} : correct / total = {} / {} = {}'.format(number, c, t, c/t))
        total += t
        correct += c
    print('Total correct / total = {} / {} = {}'.format(correct, total, correct / total))
