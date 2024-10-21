import os
import torch
import numpy as np
from preprocess import load_data
from utils import get_subset_index
import matplotlib.pyplot as plt

C, Y, X, A, n_clusters, n_input = load_data('binalpha', process_X=None)
h = 20
w = 16
max_row = 20
X = X.reshape(-1, w, h).transpose(0, 2, 1)
perm_indices = get_subset_index(torch.tensor(Y))
# perm_indices = torch.randperm(X.shape[0])[:n]


def draw_by_subgraphs(class_indices, X):
    # this cause unstable margin
    fig, axes = plt.subplots(
        max([len(indices) for indices in class_indices.values()]),
        len(class_indices),
        facecolor='grey')
    for class_label, indices in class_indices.items():
        for i, index in enumerate(indices):
            im = X[index]
            axes[i, class_label].imshow(im, cmap='gray')
            axes[i, class_label].set_aspect(h/w)
    for ax in axes.flat:
        ax.axis('off')
    fig.subplots_adjust(wspace=0.15, hspace=-0.85,
                        left=0.05, right=0.95, top=1, bottom=0)


def draw_clus_by_col(ax, class_imgs):
    space = 2
    nrows = max([len(indices) for indices in class_imgs.values()])
    ncols = len(class_imgs)
    img_mat = np.zeros((nrows*(h+space)+space, ncols*(w+space)+space))
    for class_label, indices in class_imgs.items():
        pix_y = class_label*(w+space)+space
        for i, x in enumerate(indices):
            pix_x = i*(h+space)+space
            img_mat[pix_x:pix_x+h, pix_y:pix_y+w] = 1 - x
    ax.imshow(img_mat, cmap='binary', vmax=1, vmin=0)
    ax.axis('off')


def save_in_dir_by_col(class_imgs, m, p=False):  
    for class_label, indices in class_imgs.items():
        dir = f'images/seperated/pi_{m}/'
        os.makedirs(dir, exist_ok=True)
        for i, x in enumerate(indices):
            fig = plt.figure(figsize=(1.25,1))
            ax = fig.add_subplot()
            ax.imshow(x, cmap='gray')
            ax.axis('off')
            fig.savefig(os.path.join(dir, f"{class_label}_{'p' if p else i}"),
                        pad_inches=0, bbox_inches='tight')
            plt.close(fig)
    return fig


for m in range(5):
    X_p = X[perm_indices]
    y_p = C[perm_indices, m]
    labels_p, y_p = np.unique(y_p, return_inverse=True)
    
    class_images = {}
    for i, label in enumerate(y_p):
        if label not in class_images:
            class_images[label] = []
        class_images[label].append(X_p[i])
    class_prototypes = {}
    C_m = C[:, m].reshape(-1)
    for label in class_images:
        y = labels_p[label]
        X_c = X[C_m == y]
        prototype = X_c.mean(axis=0)
        prototype = np.where(prototype>0.66, prototype, np.power(prototype, 3))
        class_prototypes[label] = [prototype]
    
    save_in_dir_by_col(class_images, m)
    save_in_dir_by_col(class_prototypes, m, True)

    # fig = plt.figure()
    # axes = fig.subplots(2, 1)
    # draw_clus_by_col(axes[1], class_images)
    # draw_clus_by_col(axes[0], class_prototypes)
    # fig.subplots_adjust(
    #     hspace=-0.6,
    #     left=0, right=1, top=1, bottom=0)
    # fig.savefig(f'results/temp{m}.png', dpi=600)
