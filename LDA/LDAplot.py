from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pickle

with open('./proj_for_plot.pkl', 'rb') as f:
    proj_for_plot = pickle.load(f)

for idx in range(10):
    sns.distplot(proj_for_plot[idx][0], bins=50)
    sns.distplot(proj_for_plot[idx][1], bins=50)
    plt.savefig('./fig/{}.png'.format(idx))
    plt.clf()
