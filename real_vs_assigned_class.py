import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from bin.common import *


parser = argparse.ArgumentParser(description='Plot results based on given table')
parser.add_argument('--subset', action='store', metavar='NAME', type=str, default='test',
                    help='Name of test subset to plot, output values will be taken from: '
                         '[PATH]/results/[NAMESPACE]/[NAMESPACE]:[SUBSET]_outputs.npy')
parser = basic_params(parser, param=True)
args = parser.parse_args()

path, output, namespace, seed = parse_arguments(args, None, model_path=True)

subset = args.subset
labels = list(np.load(os.path.join(path, '{}_{}_labels.npy'.format(namespace, subset)), allow_pickle=True).flatten())
outputs = np.load(os.path.join(path, '{}_{}_outputs.npy'.format(namespace, subset)), allow_pickle=True)
param_file = os.path.join(path, '{}_params.txt'.format(namespace))
classes = read_classes(param_file)
num_classes = len(classes)

values_per_class = [[] for _ in range(num_classes)]
real_values = []
order_real_class = [0 for _ in range(num_classes)]
for label in labels:
    for j in range(num_classes):
        value = outputs[label][j][order_real_class[label]]
        values_per_class[j].append(value)
        if j == label:
            real_values.append(value)
    order_real_class[label] += 1

num_batches = math.ceil(len(labels)/50)
for batch in range(num_batches):
    num_seqs = len(labels[50*batch:50*batch+50])
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(1, 1, 1)
    xvalues = [i for i in range(1, num_seqs+1)]
    ax.set_xticks(xvalues, minor=True)
    ax.grid(True, which='both', alpha=0.3, axis='x')
    for i, (class_name, yvalues) in enumerate(zip(classes, values_per_class)):
        plt.scatter(xvalues, yvalues[50*batch:50*batch+50], c=COLORS[i], label=class_name, alpha=0.5)
    plt.scatter([el-0.1 for el in xvalues], real_values[50*batch:50*batch+50], marker=5, c='black', label='Real class', alpha=0.7)
    plt.xlim(0, num_seqs+1)
    plt.legend(fontsize=15, bbox_to_anchor=(0, -0.32), loc="lower left", ncol=num_classes+1)
    plt.title('{}/{} - {} assigned by {}'.format(batch+1, num_batches, subset, namespace), fontsize=21)
    plt.xlabel('Sequence', fontsize=15)
    plt.ylabel('Network output', fontsize=15)
    plt.tight_layout()
    plot_file = os.path.join(output, '{}_{}_{}_real-vs-assigned.png'.format(batch+1, namespace, subset))
    plt.savefig(plot_file)
    plt.show()
    print('{} plot saved to {}'.format(batch+1, plot_file))


