from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from scipy import stats
import math

def calc_mu_sigma2(X):
    X = np.array(X)
    mu = np.mean(X)
    sigma2 = np.mean(X * X) - mu * mu
    return mu, sigma2

with open('./proj_for_plot.pkl', 'rb') as f:
    proj_for_plot = pickle.load(f)

for idx in range(10):
    ax = sns.distplot(proj_for_plot[idx][0], bins=50, label='pos')
    ax = sns.distplot(proj_for_plot[idx][1], bins=50, label='neg')
    mu_pos, sigma2_pos = calc_mu_sigma2(proj_for_plot[idx][0])
    mu_neg, sigma2_neg = calc_mu_sigma2(proj_for_plot[idx][1])
    n_pos, n_neg = len(proj_for_plot[idx][0]), len(proj_for_plot[idx][1])
    n_pos, n_neg = n_pos / (n_pos + n_neg), n_neg / (n_pos + n_neg)
    S = (mu_pos - mu_neg) ** 2 / (n_pos * sigma2_pos + n_neg * sigma2_neg)
    print("idx:{}, S={:.4f}".format(idx, S))
    # ax.set_title('LDA for number {}, S = {:.4f}'.format(idx, S))
    ax.set_title('X\\beta values for Logistic Regression, number: {}'.format(idx))
    plt.legend()
    plt.savefig('./fig/{}.png'.format(idx))
    plt.clf()

    pos_dist = stats.gaussian_kde(proj_for_plot[idx][0])
    neg_dist = stats.gaussian_kde(proj_for_plot[idx][1])
    minx, maxx = min(proj_for_plot[idx][0]), max(proj_for_plot[idx][0])
    minx, maxx = min((minx, min(proj_for_plot[idx][1]))), max((maxx, max(proj_for_plot[idx][1])))
    x = np.arange(minx-0.05*(maxx-minx), maxx+0.05*(maxx-minx),(maxx-minx) / 1000.0)
    real = lambda x: (pos_dist(x) * n_pos) / (pos_dist(x) * n_pos + neg_dist(x) * n_neg)
    sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))
    y_real = np.array([real(xx) for xx in x])
    y_sigmoid = np.array([sigmoid(xx) for xx in x])
    plt.plot(x, y_real, label='real distribution')
    plt.plot(x, y_sigmoid, label='sigmoid result')
    plt.title('real distribution VS sigmoid result, for LR model, number : {}'.format(idx))
    plt.legend()
    plt.savefig('./fig/{}-2.png'.format(idx))
    plt.clf()
