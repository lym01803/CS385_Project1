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
    filepath = './cnn.pkl'
print('loading...')
with open(filepath, 'rb') as f:
    D = pickle.load(f)
print('loaded')
X = D['X']
# X = X.reshape(-1, X.shape[1])
Y = D['Y']
X = X.numpy()
Y = Y.numpy()
print(X.shape, Y.shape)
pca = PCA(n_components=100)
# pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
tsne = TSNE(n_components=2)
X_pca = pca.fit_transform(X)
X_tsne = tsne.fit_transform(X_pca)
df = pd.DataFrame(dict(X=X_tsne[:, 0], Y=X_tsne[:, 1], Label=Y))
ax = sns.lmplot('X', 'Y', hue='Label', data=df, fit_reg=False, size=8, aspect=2)
plt.show()

