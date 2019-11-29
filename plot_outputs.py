import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from bin.funcs import basic_params, parse_arguments

parser = argparse.ArgumentParser(description='Plot results based on given table')
parser.add_argument('-f', '--file', action='store', metavar='NAME', type=str, default=[], nargs='+',
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
                neurons = line.split(':')[1].split('; ')
                break
    files = []
    for neuron in neurons:
        files.append(os.path.join(path, '{}_outputs_{}.npy'.format(namespace, neuron)))

fig, axes = plt.subplots(nrows=1, ncols=len(neurons), figsize=(15, 8), squeeze=False)
for file, ax, name in zip(files, axes, neurons):
    matrix = np.load(file)
    ax.boxplot(matrix)
    ax.set_xticks(neurons)
    ax.set_title('True label - {}'.format(name))
plt.show()
fig.savefig(os.path.join(output, '{}_outputs.png'.format(namespace)))
