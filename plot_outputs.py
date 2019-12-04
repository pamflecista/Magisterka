import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from bin.funcs import basic_params, parse_arguments
import torch

COLORS = ['C{}'.format(i) for i in range(10)]

parser = argparse.ArgumentParser(description='Plot results based on given table')
parser.add_argument('-f', '--file', action='store', metavar='NAME', type=str, default=None, nargs='+',
                    help='Files with the outputs to plot, if PATH is given, file is supposed to be '
                         'in PATH directory: [PATH]/[NAME], default: [PATH]/[NAMESPACE]_outputs.npy')
parser = basic_params(parser, plotting=True)
args = parser.parse_args()

path, output, namespace, seed = parse_arguments(args, args.file)
if args.file:
    if args.path is not None:
        file = os.path.join(path, args.file)
    else:
        file = args.file
else:
    file = os.path.join(path, '{}_outputs.npy'.format(namespace))

with open(os.path.join(path, '{}_params.txt'.format(namespace)), 'r') as f:
    for line in f:
        if line.startswith('Possible classes'):
            neurons = [el.strip().replace(' ', '-') for el in line.split(':')[1].split('; ')]
            break


def set_box_color(box, color):
    plt.setp(box['boxes'], color=color)
    plt.setp(box['whiskers'], color=color)
    plt.setp(box['caps'], color=color)
    plt.setp(box['medians'], color=color)


fig, axes = plt.subplots(nrows=len(neurons), ncols=1, figsize=(10, 15), squeeze=True, sharex=True, sharey=True)
colors = COLORS[:len(neurons)]
values = np.load(file, allow_pickle=True)
for j, (row, ax, name) in enumerate(zip(values, axes, neurons)):
    if row.any():
        for i, m in enumerate(row):
            box = ax.boxplot(m, positions=[i+1], widths=[0.6])
            set_box_color(box, colors[i])
    else:
        ax.plot([])
    ax.set_ylabel(name, color=colors[j])
    ax.set_xticks([])
    ax.set_ylim(-0.05, 1.05)
#for color, neuron in zip(colors, neurons):
 #   plt.plot([], c=color, label=neuron)
fig.suptitle(namespace, fontsize=16)
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=len(neurons))
ax = fig.add_subplot(111, frameon=False)
ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
ax.grid(False)
ax.set_ylabel("Real labels", fontsize=13)
ax.set_title('Neurons', fontsize=13)
ax.yaxis.set_label_coords(-0.1, 0.5)
plt.xticks([(i+1)*0.2 for i in range(len(neurons))], neurons)
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), colors):
    ticklabel.set_color(tickcolor)
plt.savefig(os.path.join(output, '{}_outputs.png'.format(namespace)))
plt.show()
