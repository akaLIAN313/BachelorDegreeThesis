import os
import sys
import torch
import numpy as np
from preprocess import load_data, get_similar_pairs
import matplotlib.pyplot as plt

C, Y, X, A, n_clusters, n_input = load_data(sys.argv[1])
C, Y, X, A = torch.tensor(C), torch.tensor(Y), \
    torch.tensor(X, dtype=torch.float64), torch.tensor(A)
theta_r = 1
theta_l = 0.5
theta_list = np.linspace(theta_l, theta_r, 10, endpoint=False)[::-1][:-1]
print(theta_list)
theta_ind_list = [0, 1, 2, 3]
k_list = np.array([2, 4, 8, 16])
acc_A_list = np.zeros(theta_list.shape[0])
acc_knn_list = np.zeros(k_list.shape[0])
acc_inter_mat = np.zeros((theta_list.shape[0], k_list.shape[0]))
acc_union_mat = np.zeros((theta_list.shape[0], k_list.shape[0]))
n_A_list = np.zeros(theta_list.shape[0])
n_knn_list = np.zeros(k_list.shape[0])
n_inter_mat = np.zeros((theta_list.shape[0], k_list.shape[0]))
n_union_mat = np.zeros((theta_list.shape[0], k_list.shape[0]))
for i, _theta in enumerate(theta_list):
    for j, _k in enumerate(k_list):
        (n_A, n_knn, n_inter, n_union), (acc_A, acc_knn, acc_inter, acc_union) \
            = get_similar_pairs(
                C, Y, X, A, theta=_theta, k=_k,
                print_neighbour_detail=False, print_knn_detail=False,
                return_eva=True)
        if j == 0:
            acc_A_list[i] = acc_A
            n_A_list[i] = n_A
        if i == 0:
            acc_knn_list[j] = acc_knn
            n_knn_list[j] = n_knn
        acc_inter_mat[i, j] = acc_inter
        acc_union_mat[i, j] = acc_union
        n_inter_mat[i, j] = n_inter
        n_union_mat[i, j] = n_union


fig, axes = plt.subplots(2, 4, figsize=(12, 8),
                         layout='constrained')
axes: np.ndarray[plt.Axes]
for i in range(k_list.shape[0]):
    axes[0,i].plot(theta_list, acc_A_list,
                   label=r'$\mathcal{P}_\theta$',
                   c='g', linestyle='solid', marker='s')
    axes[0,i].plot(theta_list, acc_inter_mat[:, i],
                   label=r'$\mathcal{P}_{inter}$',
                 c='r', linestyle='solid', marker='s')
    axes[0,i].plot(theta_list, acc_union_mat[:, i],
                   label=r'$\mathcal{P}_{union}$',
                   c='b', linestyle='solid', marker='s')
    axes[0,i].set_xlabel('\u03b8')
    axes[0,i].set_xlim(theta_r, theta_l)
    axes[0,i].set_xticks(theta_list[1::2])
    axes[0,i].set_ylim(0.4, 0.9)
    axes[0,i].set_yticks([0.1*i for i in range(5, 9)])
    axes[0,i].set_aspect(1)
    axes[0,i].set_title(f'k={k_list[i]}')
    if i == 0:
        axes[0,i].set_ylabel('precision')
    if i == 3:
        axes[0,i].legend()

for i in range(k_list.shape[0]):
    axes[1,i].plot(theta_list, n_A_list,
                   c='g', linestyle='solid', marker='s')
    axes[1,i].plot(theta_list, n_inter_mat[:, i],
                   c='r', linestyle='solid', marker='s')
    axes[1,i].plot(theta_list, n_union_mat[:, i],
                   c='b', linestyle='solid', marker='s')
    axes[1,i].set_xlabel('\u03b8')
    axes[1,i].set_xlim(theta_r, theta_l)
    axes[1,i].set_xticks(theta_list[1::2])
    axes[1,i].set_ylim(0, 20000)
    axes[1,i].set_aspect(0.0001/3)
    axes[1,i].set_title(f'k={k_list[i]}')
    if i == 0:
        axes[1,i].set_ylabel('number of pairs')
dir = f'images/pair/'
os.makedirs(dir, exist_ok=True)
fig.savefig(os.path.join(dir, 'precision_num'), dpi=600)
plt.close(fig)
