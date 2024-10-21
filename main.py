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
    metric_names, sim_loss, info_loss, cluster_and_evaluate
from model import HGIN
from preprocess import get_preprocessed_data


def log_best_and_draw(func):
    @wraps(func)
    def with_loggin(*args, **kwargs):
        run_time = time.strftime("%m-%d-%H:%M:%S")
        expr_name = f'{run_time}'
        if expr_args['iter_arg']:
            with open(os.path.join(dir, f'{expr_name}.txt'), 'a') as log_file:
                for arg in cur_args:
                    print(f'{arg}: {cur_args[arg]}', file=log_file)
        metrics, losses =  func(*args, **kwargs, expr_name=expr_name)
        metrics = np.array(metrics).T
        best_metric = metrics.max(axis=1)
        with open(sum_filename, 'a') as f:
            f.write(expr_name+'\n')
            line = [f'{metric_name}: {val:.4f}'
                    for metric_name, val in zip(metric_names, best_metric)]
            f.write(',  '.join(line)+'\n')
        fig, (metric_ax, loss_ax) = plt.subplots(2, 1, layout='constrained')
        for m in range(len(metric_names)):
            metric_ax.plot((range(0, expr_args['num_epochs'])), metrics[m],
                     label=metric_names[m])
        metric_ax.set_xlabel('Epochs')
        metric_ax.set_ylabel('Metric Values')
        metric_ax.legend()
        losses = np.array(losses).T
        for name, lbd, loss in zip(loss_names, lambdas, losses):
            if name == 'KL':
                kl_ax = loss_ax.twinx()
                kl_ax.plot((range(0, expr_args['num_epochs'])), loss,
                           label=name, color='r')
            else:
                loss_ax.plot((range(0, expr_args['num_epochs'])), loss,
                             label=name)
        loss_ax.set_xlabel('Epochs')
        loss_ax.set_ylabel('Loss Values')
        loss_ax.legend()
        fig.savefig(os.path.join(dir, f'{run_time}.png'), dpi=1200)
    return with_loggin


@log_best_and_draw
def main(expr_args, expr_name):
    G, meta_paths, labels, features, num_classes, n_input, \
        pairs, pair_mats = \
        get_preprocessed_data(
            expr_args['dataset'],
            process_X=expr_args['process_X'], dim_X=expr_args['dim_X'],
            theta=expr_args['theta'], k=expr_args['k'])
    pair_mats = [pair_mat.to(expr_args['device']) for pair_mat in pair_mats]
    target_node_type = 'entity'
    features = features.to(expr_args['device'], dtype=torch.float32)
    labels = labels.to(expr_args['device'])

    model = HGIN(G, features, target_node_type, meta_paths,
                 in_size=features.shape[1],
                 hidden_size=expr_args['hidden_size'],
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
            num_classes if expr_args['use_logit'] \
                else expr_args['hidden_size']),
        requires_grad=True)
    if expr_args['initialize'] == 'center':
        with torch.no_grad():
            target_embedding, _ = model(G)
            _, cluster_centers = cluster_and_evaluate(
                target_embedding, labels, num_classes)
            miu.data = torch.tensor(cluster_centers).to(expr_args['device'])
    else:
        miu.data = torch.rand(miu.data.shape).to(expr_args['device'])

    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}, {'params': miu}],
        lr=expr_args['lr'], weight_decay=expr_args['weight_decay'])
    
    with open(os.path.join(dir, f'{expr_name}.txt'), 'a') as log_file:
        metrics = []
        losses = []
        for epoch in tqdm.tqdm(range(expr_args['num_epochs'])):
            model.train()
            target_embedding, logits = model(G)
            Q = q_distribute(target_embedding, miu)
            P = target_distribution(Q)
            loss_kl = F.kl_div(torch.log(Q), P, reduction='batchmean')
            if expr_args['contrast_loss'] == 'InfoNCE':
                contrast_losses = info_loss(
                    pair_mats, lambdas[1:], target_embedding)
            else:
                contrast_losses = sim_loss(pairs, lambdas[1:], target_embedding,
                                  sim=expr_args['contrast_loss'])
            loss = expr_args['lambda_kl']*loss_kl + \
                sum([contrast_loss for contrast_loss in contrast_losses])
            print(loss_kl.item(),
                  ' '.join([f'{val}'\
                            for lbd, val in zip(lambdas[1:], contrast_losses)]),
                  file=log_file)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(
                [loss_kl.detach().cpu()] + \
                [val.detach().cpu()
                 for lbd, val in zip(lambdas[1:], contrast_losses)]
            )
            metric, _ = cluster_and_evaluate(
                target_embedding, labels, num_classes)
            line = [f'{metric_name}: {val:.4f}'
                for metric_name, val in zip(metric_names, metric)]
            print(','.join(line), file=log_file, flush=True)
            metrics.append(metric)
        return metrics, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Bachelor')
    parser.add_argument('-s', '--seed', type=int,
                        default=1, help='Random seed')
    parser.add_argument('-dev', '--device', type=str, default='cuda:0')

    parser.add_argument('--iter_arg', action='store_true')
    parser.add_argument('-d', '--dataset', type=str, default='binalpha')

    parser.add_argument('--process_X', type=str, default='origin',
                        choices=['origin', 'random', 'one-hot',
                                 'norm', 'std', 'PCA', 'A'])
    parser.add_argument('--dim_X', type=int, default=300)
    parser.add_argument('--theta', type=float, default=0.95)
    parser.add_argument('-k', type=int, default=4)
    parser.add_argument('-ini', '--initialize',
                        default='center', choices=['center', 'random'])

    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_mlp_layers', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('-lg', '--use_logit',
                        action='store_true')

    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('-lr', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--contrast_loss', type=str, default='cos',
                        choices=['InfoNCE', 'cos', 'dis'])
    parser.add_argument('-l1', '--lambda_kl', type=float, default=1)
    parser.add_argument('--lambda_neigh', type=float, default=0)
    parser.add_argument('--lambda_knn', type=float, default=0)
    parser.add_argument('-l2', '--lambda_inter', type=float, default=2**3)
    parser.add_argument('-l3', '--lambda_union', type=float, default=2**3)
    expr_args = parser.parse_args().__dict__

    
    lambdas = [expr_args['lambda_kl'],
               expr_args['lambda_neigh'], expr_args['lambda_knn'],
               expr_args['lambda_inter'], expr_args['lambda_union']]
    loss_names = ['KL'] + ['neigh', 'knn', 'inter', 'union']
    loss_list = [(lbd, name) for lbd, name in zip(lambdas, loss_names)
                 if lbd != 0]
    lambdas, loss_names = zip(*loss_list)
    
    expr_time = time.strftime("%m-%d-%H:%M")
    dir = f'logs/{expr_args["dataset"]}/{expr_time}_{expr_args["dataset"]}'
    os.makedirs(dir, exist_ok=True)
    sum_filename = os.path.join(dir, 'summary.txt')
    if expr_args['iter_arg']:
        vary_arg_dict = {
            # 'lambda_inter': [0, 2, 4, 8, 16],
            # 'lambda_union': [0, 2, 4, 8, 16],
            'k': [2, 4, 8, 16, 32],
            'theta': [0.6+i/10 for i in range(0, 4)],
        }
        vary_arg_set = vary_arg_dict.keys()
        combination_num = 1
        for arg in vary_arg_dict:
            combination_num = combination_num * len(vary_arg_dict[arg])
        val_sets = vary_arg_dict.values()
        with open(sum_filename, 'w') as f:
            fixed_arg = set(expr_args.keys()) - set(vary_arg_set)
            for arg in sorted(fixed_arg):
                print(f'{arg}: {expr_args[arg]}', file=f)
        combinations = product(*val_sets)
        for i, comb in enumerate(combinations):
            print(f'{i} of {combination_num} combinations')
            cur_args = dict(zip(vary_arg_set, comb))
            expr_args.update(cur_args)
            main(expr_args)
    else:   
        with open(sum_filename, 'w') as f:
            fixed_arg = set(expr_args.keys())
            for arg in sorted(fixed_arg):
                print(f'{arg}: {expr_args[arg]}', file=f)
        main(expr_args)
