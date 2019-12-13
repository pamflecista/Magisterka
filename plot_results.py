import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import math
from bin.funcs import basic_params, parse_arguments

COLORS = ['C{}'.format(i) for i in range(10)]

STAGES = {
    'train': 'Training',
    'val': 'Validation'
}

SHORTCUTS = {
    'loss': 'Loss',
    'sens': 'Sensitivity',
    'spec': 'Specificity',
    'auc': 'AUC'
}

parser = argparse.ArgumentParser(description='Plot results based on given table')
parser.add_argument('-t', '--table', action='store', metavar='NAME', type=str, default=None,
                    help='Results table with data to plot, if PATH is given, file is supposed to be '
                         'in PATH directory: [PATH]/[NAME], default: [PATH]/[NAMESPACE]_results.tsv')
parser = basic_params(parser, plotting=True)
parser.add_argument('-c', '--column', action='store', metavar='COL', nargs='+', type=str, default=['loss'],
                    help='Number or name of column(s) to plot, default: loss')
parser.add_argument('--train', action='store_true',
                    help='Use values from training, default values from validation are used')
parser.add_argument('--not_valid', action='store_true',
                    help='Do not print values from validation')
parser.add_argument('--print_mean', action='store_true',
                    help='Print also mean of the given data')
args = parser.parse_args()


path, output, namespace, seed = parse_arguments(args, args.table)

if args.table is not None:
    if args.path is not None:
        table = os.path.join(args.path, args.table)
    else:
        table = args.table
else:
    table = os.path.join(path, namespace + '_results.tsv')

if args.param is not None:
    if args.path is not None:
        param = os.path.join(path, args.param)
    else:
        param = args.param
else:
    param = os.path.join(path, namespace + '_params.txt')

columns = args.column

train = False
val = True
if args.train:
    train = True
if args.not_valid:
    val = False

epoch = 1
epochs = [epoch]
with open(table, 'r') as f:
    header = f.readline().strip().split('\t')
    colnum = []
    for c in columns:
        if str.isdigit(c):
            colnum.append(int(c) - 1)
        else:
            try:
                colnum.append(header.index(SHORTCUTS[c]))
            except ValueError:
                colnum += [i for i, el in enumerate(header) if SHORTCUTS[c] in el]
    stages = [el for el in STAGES.keys() if globals()[el]]
    values = [[[] for _ in colnum] for el in stages]  # for each stage and for each column
    for line in f:
        line = line.split('\t')
        if int(line[0]) > epoch:
            epoch = int(line[0])
            epochs.append(epoch)
        elif int(line[0]) < epoch:
            raise ValueError
        if line[1] in stages:
            i = stages.index(line[1])
            for j, c in enumerate(colnum):
                values[i][j].append([float(el) if el != '-' else np.nan for el in line[c].split(', ')])
ylims = [min(np.min(values), 0.0) - 0.05, max(np.max(values), 1.0) + 0.05]


def plot_one(ax, x, y, line, label):
    ax.plot(x, y, line, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylim(*ylims)


with open(param, 'r') as f:
    for line in f:
        if line.startswith('Possible classes'):
            neurons = line.split(':')[1].strip().split('; ')
            break

fig, axes = plt.subplots(nrows=len(colnum), ncols=len(stages), figsize=(12, 8), squeeze=False)
if axes.shape[1] > 1:
    num_xticks = 10
else:
    num_xticks = 20
for i, (stage, value) in enumerate(zip(stages, values)):  # for each stage
    axes[0, i].set_title(STAGES[stage])
    for j, c in enumerate(colnum):  # for each column
        a = axes[j][i]
        if i == 0:
            color = 'black'
            for n in neurons:
                if n in header[c]:
                    color = COLORS[neurons.index(n)]
            a.set_ylabel(header[c].replace('-', '-\n'), color=color)
        a.set_xticks([el for el in np.arange(1, len(epochs), math.ceil(len(epochs)/num_xticks))] + [len(epochs)])
        if len(value[j][0]) == len(neurons):  # check number of values for 1st epoch
            for k, n in enumerate(neurons):  # for each neuron
                y = [el[k] for el in value[j]]
                plot_one(a, epochs, y, '.', n)
        elif len(value[j][0]) == 1:  # or for single values
            plot_one(a, epochs, value[j], '.', 'general')
        else:
            raise ValueError
        if args.print_mean and len(value[j]) == len(neurons):
            y = [mean(el) for el in value[j]]
            plot_one(a, epochs, y, 'x', 'mean')

fig.suptitle(namespace)
plt.legend()
plt.show()
plotname = '-'.join([s[:5].lower() for s in stages]) + ':' + '-'.join([header[el][:4].lower() for el in colnum])
fig.savefig(os.path.join(output, namespace + '_{}.png'.format(plotname)))
