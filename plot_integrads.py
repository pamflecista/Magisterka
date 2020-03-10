import matplotlib.pyplot as plt
import argparse
from bin.common import *
from bin.datasets import SeqsDataset
import torch
from time import time
from bin.integrated_gradients import integrated_gradients

COLORS = ['C{}'.format(i) for i in range(10)]

parser = argparse.ArgumentParser(description='Calculate and plot integrated gradients based on given sequences and '
                                             'network')
parser.add_argument('--name', action='store', metavar='NAME', type=str, default=None,
                    help='Name of files with integrads, if PATH is given, model is supposed to be '
                         'in PATH directory, if NAMESPACE is given model is supposed to be in '
                         '[PATH]/results/[NAMESPACE]/ directory')
parser.add_argument('--seq', action='store', metavar='DATA', type=str, required=True,
                    help='File or folder with sequences to check, if PATH is given, file is supposed to be in '
                         '[PATH]/data/integrads/ directory.')
parser = basic_params(parser, plotting=True)
args = parser.parse_args()



fig, axes = plt.subplots(nrows=len(seq_ids), ncols=len(classes), figsize=(12, 8), squeeze=False, sharex='col',
                         sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
for i, name in enumerate(classes):
    for j, seq in enumerate(seq_ids):
        ax = axes[j, i]
        if j == 0:
            ax.set_title(name, fontsize=15)
        if i == 0:
            ax.set_ylabel(seq, fontsize=15)
        result = results[name][j]
        result = [result[:, i:i+leap].flatten() for i in range(0, seq_len, leap)]
        ax.boxplot(result)
        labels = [1] + [i for i in np.arange(0, seq_len + 1,  seq_len//4)][1:]
        xticks = [i for i in range(0, len(result)+1, len(result)//4)]
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(-10, 10)
fig.suptitle('Integrated gradients - {}'.format(analysis_name), fontsize=15)
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(output, namespace + 'integrads_{}_{}.png'.format(analysis_name, seq_name)))
