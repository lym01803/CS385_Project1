import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import random
import pickle

import torch
import torch.nn as nn
from torch.nn import Embedding

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE

import matplotlib as mpl

import pickle, os, sys

sys.path.append('..')
from utils import *

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
Y = D['train']['label']
Y2 = label2onehot(Y)

print(X.shape, Y.shape)
X = X[:5000]
Y = Y[:5000]
Y2 = Y2[:5000]
#pca = PCA(n_components=100)
pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
# tsne = TSNE(n_components=2)
X_pca = pca.fit_transform(X)
# X_tsne = tsne.fit_transform(X_pca)
df = pd.DataFrame(dict(X=X_pca[:, 0], Y=X_pca[:, 1], Label=Y2[:, 2]))
ax = sns.lmplot('X', 'Y', hue='Label', data=df, fit_reg=False, size=8, aspect=2)
plt.show()

