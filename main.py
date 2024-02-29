import os
import sys
import time
from functools import wraps
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import tqdm
from utils import q_distribute, target_distribution, \
    sim_loss, cluster_and_evaluate
from model import HGIN
from preprocess import get_preprocessed_data


def store_best_result_and_fig(func):
    @wraps(func)
    def with_loggin(*args, **kwargs):
        run_time = time.strftime("%m-%d-%H:%M:%S")
        expr_name = f'{run_time}_{expr_args["dataset"]}_{expr_args["process_X"]}'
        metrics, losses =  func(*args, **kwargs, expr_name=expr_name)
        metrics = np.array(metrics).T
        best_metric = metrics.max(axis=1)
        with open(sum_filename, 'a') as f:
            f.write(expr_name+'\n')
            line = [f'{metric_name}: {val:.4f}'
                    for metric_name, val in zip(metric_names, best_metric)]
            f.write(',  '.join(line)+'\n')
        fig, (metric_ax, loss_ax) = plt.subplots(2, 1)
        for m in range(len(metric_names)):
            metric_ax.plot((range(0, expr_args['num_epochs'])), metrics[m],
                     label=metric_names[m])
            metric_ax.set_xlabel('Epochs')
            metric_ax.set_ylabel('Metric Values')
            metric_ax.legend()
        losses = np.array(losses).T
        for m in range(len(loss_names)):
            loss_ax.plot((range(0, expr_args['num_epochs'])), losses[m],
                     label=loss_names[m])
            loss_ax.set_xlabel('Epochs')
            loss_ax.set_ylabel('Loss Values')
            loss_ax.legend()
        fig.savefig(os.path.join(dir, f'{run_time}.png'))
    return with_loggin


@store_best_result_and_fig
def main(expr_args, expr_name):
    G, meta_paths, labels, features, num_classes, n_input, \
        neighbouring_pairs, knn_pairs, inter, union = \
        get_preprocessed_data(
            expr_args['dataset'], process_X=expr_args['process_X'],
            dim_X=expr_args['dim_X'], theta=expr_args['theta'], k=expr_args['k'])
    target_node_type = 'entity'
    features = features.to(expr_args['device'], dtype=torch.float32)
    labels = labels.to(expr_args['device'])

    model = HGIN(G, features, target_node_type, meta_paths,
                 in_size=features.shape[1], hidden_size=expr_args['hidden_size'],
                 out_size=num_classes,
                 num_layers=expr_args['num_layers'],
                 num_mlp_layers=expr_args['num_mlp_layers'],
                 dropout=expr_args['dropout'], device=expr_args['device']
                 ).to(expr_args['device'])
    for p in model.parameters():
        if len(p.data.shape) > 1:
            nn.init.xavier_uniform_(p.data)
    miu = nn.Parameter(
        torch.Tensor(
            num_classes,
            num_classes if expr_args['use_logit'] else expr_args['hidden_size']),
        requires_grad=True)
    if expr_args['initialize'] == 'center':
        with torch.no_grad():
            target_embedding, _ = model(G)
            _, _, _, _, cluster_centers = cluster_and_evaluate(
                target_embedding, labels, num_classes)
            miu.data = torch.tensor(cluster_centers).to(expr_args['device'])
    else:
        miu.data = torch.rand(miu.data.shape).to(expr_args['device'])

    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}, {'params': miu}],
        lr=expr_args['lr'], weight_decay=expr_args['weight_decay'])
    
    with open(os.path.join(dir, f'{expr_name}.txt'), 'w') as log_file:
        metrics = []
        losses = []
        for epoch in tqdm.tqdm(range(expr_args['num_epochs'])):
            model.train()
            target_embedding, logits = model(G)
            Q = q_distribute(target_embedding, miu)
            P = target_distribution(Q)
            loss_kl = F.kl_div(torch.log(Q), P, reduction='batchmean')
            loss_inter = sim_loss(inter, target_embedding)
            loss_union = sim_loss(union, target_embedding)
            loss = expr_args['lambda_kl']*loss_kl + \
                expr_args['lambda_inter']*loss_inter + \
                expr_args['lambda_union']*loss_union
            print(loss_kl.item(), loss_inter.item(), loss_union.item(),
                  file=log_file)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append([loss_kl.detach().cpu(),
                           loss_inter.detach().cpu(), loss_union.detach().cpu()])
            acc, nmi, ari, f1, _ = cluster_and_evaluate(
                target_embedding, labels, num_classes)
            print("ACC: {:.4f},".format(acc), "NMI: {:.4f},".format(nmi),
                "ARI: {:.4f},".format(ari), "F1: {:.4f}".format(f1),
                file=log_file)
            metrics.append([acc, nmi, ari, f1])
        return metrics, losses


if __name__ == "__main__":
    metric_names = ['ACC', 'NMI', 'ARI', 'F1']
    loss_names = ['KL', 'inter', 'union']

    parser = argparse.ArgumentParser('Bachelor')
    parser.add_argument('-s', '--seed', type=int,
                        default=1, help='Random seed')
    parser.add_argument('-dev', '--device', type=str, default='cuda:1')

    parser.add_argument('--iter_arg', action='store_true')
    parser.add_argument('-d', '--dataset', type=str, default='binalpha')
    parser.add_argument('--process_X', type=str, default='origin',
                        choices=['origin', 'random', 'one-hot',
                                 'norm', 'std', 'PCA', 'A'])
    parser.add_argument('--dim_X', type=int, default=300)
    parser.add_argument('-p', type=str, default='inter',
                        choices=['inter', 'union', 'knn', 'neigubour'])
    parser.add_argument('--theta', type=float, default=0.95)
    parser.add_argument('-k', type=int, default=4)
    parser.add_argument('-ini', '--initialize',
                        default='center', choices=['center', 'random'])

    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_mlp_layers', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('-lg', '--use_logit',
                        action='store_true')

    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('-l1', '--lambda_kl', type=float, default=1)
    parser.add_argument('-l2', '--lambda_inter', type=float, default=2**3)
    parser.add_argument('-l3', '--lambda_union', type=float, default=2**3)
    expr_args = parser.parse_args().__dict__
    config_set = {
        'lr': [0.1**i for i in range(3, 5)],
        'lambda_inter': [2**i for i in range(4, 7, 2)],
        'lambda_union': [2**i for i in range(0, 5, 2)],
        'k': [2**i for i in range(1, 6, 2)],
        'hidden_size': [2**i for i in range(7, 10)]
    }
    arg_set = config_set.keys()
    val_sets = config_set.values()
    combinations = product(*val_sets)
    expr_time = time.strftime("%m-%d-%H:%M")
    dir = f'logs/{expr_time}_{expr_args["dataset"]}'
    os.mkdir(dir)
    sum_filename = os.path.join(dir, 'summary.txt')
    if expr_args['iter_arg']:
        with open(sum_filename, 'w') as f:
            fix_arg = set(expr_args.keys()) - set(arg_set)
            for arg in fix_arg:
                print(f'{arg}: {expr_args[arg]}', file=f)
        for comb in combinations:
            config = dict(zip(arg_set, comb))
            expr_args.update(config)
            main(expr_args)
    else:
        with open(sum_filename, 'w') as f:
            fix_arg = set(expr_args.keys())
            for arg in fix_arg:
                print(f'{arg}: {expr_args[arg]}', file=f)
        main(expr_args)
