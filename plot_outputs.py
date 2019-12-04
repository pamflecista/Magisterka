import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from bin.funcs import basic_params, parse_arguments
import torch

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

parser = argparse.ArgumentParser(description='Plot results based on given table')
parser.add_argument('-f', '--file', action='store', metavar='NAME', type=str, default=None, nargs='+',
                    help='Files with the outputs to plot, if PATH is given, file is supposed to be '
                         'in PATH directory: [PATH]/[NAME], default: [PATH]/[NAMESPACE]_outputs_{}.npy')
parser = basic_params(parser, plotting=True)
args = parser.parse_args()

path, output, namespace, seed = parse_arguments(args, args.file)
if args.file:
    if args.path is not None:
        files = [os.path.join(path, file) for file in args.file]
    else:
        files = args.file
else:
    if args.param is not None:
        param = args.param.split('/')[-1]
    else:
        param = '{}_params.txt'.format(namespace)
    with open(os.path.join(path, param), 'r') as f:
        for line in f:
            if line.startswith('Possible classes'):
                neurons = [el.strip().replace(' ', '-') for el in line.split(':')[1].split('; ')]
                break
    files = []
    for neuron in neurons:
        files.append(os.path.join(path, '{}_outputs_{}.npy'.format(namespace, neuron)))


def set_box_color(box, color):
    plt.setp(box['boxes'], color=color)
    plt.setp(box['whiskers'], color=color)
    plt.setp(box['caps'], color=color)
    plt.setp(box['medians'], color=color)


fig, axes = plt.subplots(nrows=1, ncols=len(neurons), figsize=(15, 8), squeeze=True)
colors = COLORS[:len(neurons)]
for j, (file, ax, name) in enumerate(zip(files, axes, neurons)):
    matrix = np.load(file, allow_pickle=True)
    if matrix.any():
        for i, m in enumerate(matrix):
            box = ax.boxplot(m, positions=[i+1], widths=[0.6])
            set_box_color(box, colors[i])
    else:
        ax.plot([])
    if j == 0:
        ax.set_ylabel('Output value')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([])
    ax.set_title(format(name), color=colors[j])
for color, neuron in zip(colors, neurons):
    plt.plot([], c=color, label=neuron)
fig.suptitle(namespace)
plt.legend(loc='upper right', bbox_to_anchor=(0.3, -0.05),
           fancybox=True, shadow=True, ncol=len(neurons))
plt.savefig(os.path.join(output, '{}_outputs.png'.format(namespace)))
plt.show()
