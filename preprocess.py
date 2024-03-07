import sys
import dgl
import numpy as np
import torch
import scipy.io as sio
from sklearn.decomposition import PCA
from utils import cluster_and_evaluate, metric_names


def load_data(name, process_X='PCA', dim_X=None):
    name = "./data/" + name
    pool_data = sio.loadmat(name + "_pool_half_vary_k.mat")
    C = pool_data["members"].astype(int)
    n_input = C.shape[0]
    Y = pool_data["gt"].astype(int).reshape(-1)
    n_clusters = np.unique(Y).shape[0]
    A = get_corelation_matrix(C, False)
    raw_data = sio.loadmat(name + "_data.mat")
    X = raw_data["data"]
    assert C.shape[0] == X.shape[0]
    if process_X == 'PCA' or X.shape[1] > 10000:
        pca = PCA(n_components=dim_X)
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
    n_pair = torch.sum(mat)
    chance_same_label = n_same_label.float() / n_pair
    print(f'pairs: {n_pair.item()}, {chance_same_label.item():.3f}')


def get_similar_pairs(C, Y, X, A, soft_A=False, theta=0.95, k=4,
                      print_neighbour_detail=False, print_knn_detail=True):
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
                f'{threshold:.2f}, {n_same_label.item()},',
                f'{n_pairs_above_threshold.item():.3f}')
    neighbouring_mat = A.fill_diagonal_(0).triu() > theta
    verify_pairs(coclass_mat, neighbouring_mat)
    neighbouring_pairs = torch.where(neighbouring_mat)
    neighbouring_pairs = torch.stack(neighbouring_pairs)
    # similarity = np.matmul(X, X.T)/np.float_power(np.linalg.norm(X, axis=1), 2)
    distances = torch.norm(X[:, np.newaxis, :] - X, dim=2)
    similarity = torch.exp(-1*distances).fill_diagonal_(0)
    sorted_indices = torch.argsort(similarity, dim=1).flip([1])
    if print_knn_detail:
        for _k in range(1, 20, 1):
            num_same_label = torch.sum(Y.unsqueeze(1) == Y[sorted_indices[:, :_k]])
            print(f'{_k:2d} {num_same_label.item():8d}',
                  f'{num_same_label.item()/(_k*n_input):.4f}',
                  end='\n' if _k%5==4 else '\t')
    k_neighbours = sorted_indices[:, :k].reshape(-1)
    nodes = torch.repeat_interleave(torch.arange(n_input), k).reshape(-1)
    knn_pairs = torch.stack([nodes, k_neighbours])
    knn_mat = torch.zeros((n_input, n_input))
    knn_mat[nodes, k_neighbours] = 1
    knn_mat = (knn_mat > 0).triu()
    verify_pairs(coclass_mat, knn_mat)
    inter_mat = knn_mat & neighbouring_mat
    verify_pairs(coclass_mat, inter_mat)
    inter = torch.stack(torch.where(inter_mat))
    union_mat = knn_mat | neighbouring_mat
    verify_pairs(coclass_mat, union_mat)
    union = torch.stack(torch.where(union_mat))
    return (neighbouring_pairs, knn_pairs, inter, union), \
        (neighbouring_mat, knn_mat, inter_mat, union_mat)


def get_preprocessed_data(name, process_X=None, dim_X=None, theta=0.95, k=4,
                          diff_results_type=True, raw_cluster=False):
    C, Y, X, A, n_clusters, n_input = load_data(
        name, process_X=process_X, dim_X=dim_X)
    print("{} nodes, {} clusters".format(C.shape[0], n_clusters))
    n_clusters = np.unique(Y).shape[0]
    G, metapaths = construct_dgl_graph(C, X)
    C, Y, X, A = torch.tensor(C), torch.tensor(Y), \
        torch.tensor(X, dtype=torch.float64), torch.tensor(A)
    metric, _ = cluster_and_evaluate(
        X, Y, torch.unique(Y).shape[0])
    line = ','.join([f'{metric_name}: {val:.4f}' 
                     for metric_name, val in zip(metric_names, metric)])
    print(line)
    pairs, pair_mats = get_similar_pairs(
        C, Y, X, A, theta=theta, k=k)
    if process_X == 'random':
        X = torch.rand((n_input, dim_X))
    elif process_X == 'one-hot':
        X = torch.diag(torch.ones((n_input)))
    elif process_X == 'A':
        X = A.copy()
    return G, metapaths, Y, X, n_clusters, n_input, pairs, pair_mats


if __name__ == "__main__":
    get_preprocessed_data(sys.argv[1], process_X=sys.argv[2])
