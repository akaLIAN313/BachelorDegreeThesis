import os
import glob
import argparse
parser = argparse.ArgumentParser('LogDrawer')
parser.add_argument('-d', '--dataset', type=str, default='binalpha')
parser.add_argument('-t', '--time', type=str, required=True)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('-l1', '--lambda_kl', type=float, default=1)

draw_args = parser.parse_args()
import matplotlib.pyplot as plt

# Read the log file

file_pattern = os.path.join('logs',
                         draw_args.dataset,
                         f'{draw_args.time}_{draw_args.dataset}',
                         f'{draw_args.time}*.txt')
file_path = glob.glob(file_pattern)[0]
with open(file_path, "r") as file:
    lines = file.readlines()

# Parse the log data
loss_values = []
metrics = {'ACC': [], 'NMI': [], 'PUR': [], 'ARI': [], 'F1': []}
lambda_list = [draw_args.lambda_kl, 1, 1]
for i in range(0, draw_args.num_epochs*2, 2):
    loss_line = lines[i].strip().split()
    weighted_loss = sum([float(loss) * weight
                         for loss, weight in zip(loss_line, lambda_list)])
    loss_values.append(weighted_loss)

    metric_line = lines[i + 1].strip().split(',')
    for metric in metric_line:
        key, value = metric.split(':')
        metrics[key.strip()].append(float(value))

# Plotting
fig, ax1 = plt.subplots(figsize=(6,3))

# Plot loss values
ax1.plot(loss_values, label=r'$\mathcal{L}$')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.legend()
plt.tight_layout()
plt.savefig(f'images/{draw_args.dataset}_L.png')
plt.close(fig)

fig, ax2 = plt.subplots(figsize=(6,3))
# Plot metrics
for metric, values in metrics.items():
    if metric == 'F1':
        continue
    ax2.plot(values, label=metric)
ax2.set_title('Metrics')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Value')
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig(f'images/{draw_args.dataset}_m.png')