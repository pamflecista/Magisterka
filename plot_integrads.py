import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from bin.common import *
from bin.datasets import SeqsDataset
import torch
from time import time
from bin.integrated_gradients import integrated_gradients

COLORS = ['C{}'.format(i) for i in range(10)]

parser = argparse.ArgumentParser(description='Calculate and plot integrated gradients based on given sequences and '
                                             'network')
parser.add_argument('--model', action='store', metavar='NAME', type=str, default=None,
                    help='File with the model to check, if PATH is given, model is supposed to be in PATH directory, '
                         'if NAMESPACE is given model is supposed to be in [PATH]/results/[NAMESPACE]/ directory')
parser.add_argument('--seq', action='store', metavar='DATA', type=str, required=True,
                    help='File or folder with sequences to check, if PATH is given, file is supposed to be in '
                         '[PATH]/data/integrads/ directory.')
parser = basic_params(parser, plotting=True)
args = parser.parse_args()

path, output, namespace, seed = parse_arguments(args, args.model)

if args.model is None:
    model_file = 'results/{}/{}_last.model'.format(namespace, namespace)
else:
    model_file = args.model

if args.param is None:
    param_file = 'results/{}/{}_params.txt'.format(namespace, namespace)
else:
    param_file = args.param

if path:
    seq_file = os.path.join(path, 'data/integrads/', args.seq)
    model_file = os.path.join(path, model_file)
    param_file = os.path.join(path, param_file)
else:
    seq_file = args.seq

# CUDA for PyTorch
use_cuda, device = check_cuda(None)

network, _, seq_len, _, classes = params_from_file(param_file)

dataset = SeqsDataset(seq_file, seq_len=seq_len)
seq_ids = dataset.IDs
X, y = dataset.__getitem__(0)
labels = [y]
X = [X]
for i in range(1, len(dataset)):
    xx, yy = dataset.__getitem__(i)
    X.append(xx)
    labels.append(yy)
X = torch.stack(X, dim=0)

t0 = time()
# Build network
model = network(seq_len)
# Load weights from the file
model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
print('\nModel from {} loaded in {:.2f} s'.format(model_file, time() - t0))

results = {}
leap = 10
for i, name in enumerate(classes):
    r = integrated_gradients(model, X, i)
    r = np.squeeze(r, axis=1)
    results[name] = np.array([sum(r[j:j+leap]) for j in range(0, len(r), leap)])

fig, axes = plt.subplots(nrows=len(seq_ids), ncols=len(classes), figsize=(12, 8), squeeze=False)
for i, name in enumerate(classes):
    for j, seq in enumerate(seq_ids):
        ax = axes[j, i]
        if j == 0:
            ax.set_title(name, fontsize=15)
        if i == 0:
            ax.set_ylabel(seq_ids, fontsize=15)
        result = results[name][j]
        ax.boxplot(result)
fig.suptitle('Importance of ')

