from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pickle

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
    ax.set_title('LDA for number {}, S = {:.4f}'.format(idx, S))
    plt.legend()
    plt.savefig('./fig/{}.png'.format(idx))
    plt.clf()
