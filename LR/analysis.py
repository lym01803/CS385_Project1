from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from scipy import stats
import math

with open('./proj_for_plot_rd0.01.pkl', 'rb') as f:
    proj_for_plot = pickle.load(f)

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

XWp = np.array(proj_for_plot[0][0])
XWn = np.array(proj_for_plot[0][1])
Ep = []
En = []
for i in range(XWp.shape[0]):
    xw = XWp[i]
    Ep.append(1 - sigmoid(xw))
for i in range(XWn.shape[0]):
    xw = XWn[i]
    En.append(0 - sigmoid(xw))
Ep = np.array(Ep)
En = np.array(En)
sum1 = np.sum(XWp * Ep)
sum2 = np.sum(XWn * En)
print((sum1 + sum2) / (XWp.shape[0] + XWn.shape[0]))
