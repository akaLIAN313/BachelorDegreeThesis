import sys
import dgl
import numpy as np
import torch
import scipy.io as sio
from sklearn.decomposition import PCA
from utils import cluster_and_evaluate


def load_data(name, process_X='PCA', dim_X=None):
    name = "./data/" + name
    pool_data = sio.loadmat(name + "_pool_half_vary_k.mat")
    C = pool_data["members"].astype(int)
    n_input = C.shape[0]
    Y = pool_data["gt"].astype(int).reshape(-1)
    n_clusters = np.unique(Y).shape[0]
    A = get_corelation_matrix(C, False)
    if process_X == 'random':
        X = np.random.random((n_input, dim_X))
    elif process_X == 'one-hot':
        assert n_input < 5000
        X = np.diag(np.ones((n_input)))
    elif process_X == 'A':
        assert n_input < 5000
        X = A.copy()
    else:
        raw_data = sio.loadmat(name + "_data.mat")
        X = raw_data["data"]
        assert C.shape[0] == X.shape[0]
        if process_X == 'PCA':
            pca = PCA(n_components=200)
            X = pca.fit_transform(X)
        elif process_X == 'norm':
            niu = np.mean(X, axis=0)
            theta = np.std(X, axis=0)
            theta[theta == 0] = 0.1
            X = (X-niu)/theta
        elif process_X == 'std':
            niu = np.max(X, axis=0)
            theta = niu - np.min(X, axis=0)
            theta[theta == 0] = 0.1
            X = (X-niu)/theta
    print('dim_X:', X.shape[1])
    return C, Y, X, A, n_clusters, n_input


def get_corelation_matrix(C, self_loop=False) -> np.ndarray:
    num_pre_clustering = C.shape[1]
    print("{} kinds of preclustering result".format(num_pre_clustering))
    A = np.eye(C.shape[0], dtype=int) * num_pre_clustering if self_loop\
        else np.zeros((C.shape[0], C.shape[0]))
    for i in range(C.shape[0]):
        for j in range(i+1, C.shape[0]):
            count_of_same_cluster = np.sum(C[i] == C[j])
            A[i][j] = A[j][i] = count_of_same_cluster
    A = A.astype(float)
    A /= num_pre_clustering
    return A


def construct_dgl_graph(C, X):
    n_input, n_clustering_results = C.shape
    edge_dict = {('entity', f'{i}-th belongs to', f'{i}-th clustering'):
                 (np.arange(n_input), C[:, i])
                 for i in range(n_clustering_results)}
    edge_dict_reverse = {(f'{i}-th clustering', f'{i}-th contains', 'entity'):
                         (C[:, i], np.arange(n_input))
                         for i in range(n_clustering_results)}
    edge_dict = {**edge_dict, **edge_dict_reverse}
    meta_paths = [[f'{i}-th belongs to', f'{i}-th contains']
                  for i in range(n_clustering_results)]
    G = dgl.heterograph(edge_dict)
    return G, meta_paths


def verify_pairs(coclass_mat, mat):
    n_same_label = torch.sum(mat & coclass_mat)
    chance_same_label = n_same_label.float() / torch.sum(mat)
    print(f'pairs: {n_same_label.item()}, {chance_same_label.item():.3f}')


def get_similar_pairs(C, Y, X, A, soft_A=False, theta=0.95, k=4,
                      print_neighbour_detail=True, print_knn_detail=False):
    n_input, n_clustering_results = C.shape
    assert n_input == X.shape[0]
    if soft_A:
        A = torch.softmax(A, dim=1)
    Y = Y.reshape(-1)
    coclass_mat = (Y.unsqueeze(0) == Y.unsqueeze(1)).fill_diagonal_(0)
    if print_neighbour_detail:
        for threshold in np.linspace(0, 1, 21):
            above_threshold = (A.fill_diagonal_(0).triu() > threshold)
            n_same_label = torch.sum(
                above_threshold &
                coclass_mat.triu())
            n_pairs_above_threshold = torch.sum(above_threshold)
            chance_same_label = n_same_label.float() / n_pairs_above_threshold.float()
            print(
                f'{threshold:.2f}, {n_same_label.item()}, {chance_same_label.item():.3f}')
    neighbouring_mat = A.fill_diagonal_(0).triu() > theta
    verify_pairs(coclass_mat, neighbouring_mat)
    neighbouring_pairs = torch.where(neighbouring_mat)
    neighbouring_pairs = torch.stack(neighbouring_pairs)
    # similarity = np.matmul(X, X.T)/np.float_power(np.linalg.norm(X, axis=1), 2)
    distances = torch.norm(X[:, np.newaxis, :] - X, dim=2)
    similarity = torch.exp(-1*distances).fill_diagonal_(0)
    sorted_indices = torch.argsort(similarity, dim=1).flip([1])
    if print_knn_detail:
        # could be improve by broadcast operation
        for _k in range(1, 20, 1):
            num_same_label = 0
            for i in range(n_input):
                for j in range(_k):
                    if Y[i] == Y[sorted_indices[i][j]]:
                        num_same_label += 1
            print(num_same_label/(_k*n_input))
    k_neighbours = sorted_indices[:, :k].reshape(-1)
    nodes = torch.repeat_interleave(torch.arange(n_input), k).reshape(-1)
    knn_pairs = torch.stack([nodes, k_neighbours])
    knn_mat = torch.zeros((n_input, n_input))
    knn_mat[nodes, k_neighbours] = 1
    knn_mat = (knn_mat > 0).triu()
    verify_pairs(coclass_mat, knn_mat)
    inter = knn_mat & neighbouring_mat
    verify_pairs(coclass_mat, inter)
    inter = torch.stack(torch.where(inter))
    union = knn_mat | neighbouring_mat
    verify_pairs(coclass_mat, union)
    union = torch.stack(torch.where(union))
    return neighbouring_pairs, knn_pairs, inter, union


def get_preprocessed_data(name, process_X=None, dim_X=None, theta=0.95, k=4,
                          diff_results_type=True, raw_cluster=False):
    C, Y, X, A, n_clusters, n_input = load_data(
        name, process_X=process_X, dim_X=dim_X)
    print("{} nodes, {} clusters".format(C.shape[0], n_clusters))
    n_clusters = np.unique(Y).shape[0]
    G, metapaths = construct_dgl_graph(C, X)
    C, Y, X, A = torch.tensor(C), torch.tensor(Y), \
        torch.tensor(X, dtype=torch.float64), torch.tensor(A)
    acc, nmi, ari, f1, _ = cluster_and_evaluate(
        X, Y, torch.unique(Y).shape[0])
    print("raw input clustering result:",
          "ACC: {:.4f},".format(acc), "NMI: {:.4f},".format(nmi),
          "ARI: {:.4f},".format(ari), "F1: {:.4f}".format(f1))
    neighbouring_pairs, knn_pairs, inter, union = get_similar_pairs(
        C, Y, X, A, theta=theta, k=k)
    return G, metapaths, Y, X, n_clusters, n_input, neighbouring_pairs, knn_pairs, inter, union


if __name__ == "__main__":
    get_preprocessed_data(sys.argv[1], process_X=sys.argv[2])
