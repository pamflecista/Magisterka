import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from bin.common import *

COLORS = ['C{}'.format(i) for i in range(10)]

parser = argparse.ArgumentParser(description='Compare network outputs for reference and for alternative sequences')
parser.add_argument('--name', '--test_namespace', action='store', metavar='NAME', type=str, default='test',
                    help='Namespace of test analysis, default: test')
parser = basic_params(parser)
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, args.name, model_path=True)

name = args.name.replace('_', '-')
outputs = np.load(os.path.join(path, '{}_{}_outputs.npy'.format(namespace, name)))
labels = list(np.load(os.path.join(path, '{}_{}_labels.npy'.format(namespace, name))))
seq_ids = open(os.path.join(path, '{}_{}.txt'.format(namespace, name)), 'r').read().strip().split('\n')
num_seqs = len(seq_ids)
seq_file = None
with open(os.path.join(path, '{}_test_results.tsv'.format(namespace)), 'r') as f:
    f.readline()
    for line in f:
        line = line.strip().split('\t')
        if line[1] == name:
            seq_file = line[0]
            break

label_names = ['' for _ in range(num_seqs)]
if os.path.isfile(seq_file):
    seqs = ['' for _ in range(num_seqs)]
    patients = ['' for _ in range(num_seqs)]
    with open(seq_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                l = line.strip('>\n ').split(' ')
                id = '{}:{}'.format(l[0].lstrip('chr'), l[1])
                pos = seq_ids.index(id)
                label_names[pos] = '{} {}'.format(l[3], l[4])
                patients[pos] = l[1]
            else:
                if l[-1] == 'ref':
                    ref_seq = line.strip()
                seqs[pos] = line.strip()
    num_snp = [max(len([a for a, r in zip(seq, ref_seq) if a != r]), 1) for seq in seqs]
else:
    patients = seq_ids
    num_snp = {pat: 1 for pat in patients}

classes = get_classes_names(os.path.join(path, '{}_params.txt'.format(namespace)))
xvalues = {'True class': [], 'False class': []}
yvalues = {'True class': [], 'False class': []}
sizes = {'True class': [], 'False class': []}

for i, (label, n) in enumerate(zip(labels, label_names)):
    output = outputs[label]
    seq_pos = len([el for el in labels[:i] if el == label])
    xvalues['True class'].append(label * num_seqs + i + label + 1)
    yvalues['True class'].append(output[label][seq_pos])
    sizes['True class'].append(num_snp[i])
    for wrong_name in [el for el in classes if el != n]:
        wrong_label = classes.index(wrong_name)
        xvalues['False class'].append(wrong_label * num_seqs + i + wrong_label + 1)
        yvalues['False class'].append(output[wrong_label][seq_pos])
        sizes['False class'].append(num_snp[i])

plt.figure(figsize=(15, 10))
for legend_label, color, marker in zip(['True class', 'False class'], ['green', 'red'], ['*', 'o']):
    plt.scatter(xvalues[legend_label], yvalues[legend_label], s=sizes[legend_label], color=color, marker=marker,
                label=legend_label)
xticks = [la for el in xvalues.values() for la in el]
xticks.sort()
plt.xticks(xticks, patients*len(classes), fontsize=12, rotation=45, ha='right')
plt.xlabel(('   ' * num_seqs).join(classes), fontsize=16)
plt.ylabel('Output value', fontsize=16)
plt.legend(fontsize=12)
plt.title('{} - {}'.format(namespace, name), fontsize=20)
plt.tight_layout()
plt.show()
