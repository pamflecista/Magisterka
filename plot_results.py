import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

STAGES = {
    'train': 'Training',
    'valid': 'Validation'
}

parser = argparse.ArgumentParser(description='Plot results based on given table')
parser.add_argument('-t', '--table', action='store', metavar='NAME', type=str, default=None,
                    help='Results table with data to plot, if PATH is given, file is supposed to be '
                         'in PATH directory: [PATH]/[NAME], default: [PATH]/[NAMESPACE]_results.tsv')
parser.add_argument('--namespace', action='store', metavar='NAME', type=str, default=None,
                    help='Namespace of the analysis, default: established based on [TABLE]')
parser.add_argument('-p', '--path', action='store', metavar='DIR', type=str, default=None,
                    help='Working directory.')
parser.add_argument('-o', '--output', action='store', metavar='DIR', type=str, default=None,
                    help='Output directory, default: [PATH]/results/[NAMESPACE]')
parser.add_argument('-c', '--column', action='store', metavar='NUM', nargs='+', type=int, default=[3],
                    help='Column(s) to plot, default: 3')
parser.add_argument('--train', action='store_true',
                    help='Use values from training, default values from validation are used')
parser.add_argument('--seed', action='store', metavar='NUMBER', type=int, default='0',
                    help='Set random seed, default: 0')
args = parser.parse_args()

if args.namespace is not None:
    namespace = args.namespace
elif args.table is not None:
    namespace = args.table.split('/')[-1].strip('_results.tsv')
else:
    namespace = args.path.split('/')[-1].strip('_results.tsv')

if args.table is not None:
    table = args.table.split('/')[-1]
else:
    table = os.path.join(args.path, namespace + '_results.tsv')

if args.path is not None:
    path = args.path
else:
    path = '/'.join(args.table.split('/')[:-1])

if args.output is not None:
    output = args.output
else:
    output = path

columns = [el - 1 for el in args.column]

train = False
valid = True
if args.train and not args.valid:
    train = True
    valid = False

stages = [el for el in STAGES.keys() if globals()[el]]
values = [[[] for _ in columns] for el in stages]

epoch = 1
epochs = [epoch]
with open(os.path.join(path, table), 'r') as f:
    header = f.readline().split('\t')
    for line in f:
        line = line.split('\t')
        if int(line[0]) > epoch:
            epoch = int(line[0])
            epochs.append(epoch)
        elif int(line[0]) < epoch:
            raise ValueError
        if line[1] in stages:
            i = stages.index(line[1])
            for j, c in enumerate(columns):
                values[i][j].append([float(el) if el != '-' else np.nan for el in line[c].split(', ')])


def plot_one(ax, x, y, line):
    ax.plot(x, y, line)
    ax.set_xlabel('Epoch')


fig, axes = plt.subplots(nrows=len(columns), ncols=len(stages), figsize=(12, 8), squeeze=False)
for i, (stage, value) in enumerate(zip(stages, values)):
    axes[i, 0].set_title(STAGES[stage])
    for j, c in enumerate(columns):
        a = axes[j][i]
        a.set_ylabel(header[c])
        a.set_xticks(epochs)
        neurons = len(value[j][0])
        x = [mean(el) for el in value[j]]
        plot_one(a, epochs, x, '.-')
        for n in range(neurons):
            x = [el[n] for el in value[j]]
            plot_one(a, epochs, x, '.--')

plt.show()
plotname = '-'.join([s[:5].lower() for s in stages]) + ':' + '-'.join([header[el][:4].lower() for el in columns])
fig.savefig(os.path.join(path, namespace + '_{}.png'.format(plotname)))
