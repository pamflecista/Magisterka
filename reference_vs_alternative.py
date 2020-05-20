import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from bin.common import *

COLORS = ['C{}'.format(i) for i in range(10)]

parser = argparse.ArgumentParser(description='Compare network outputs for reference and for alternative sequences')
parser.add_argument('--name', '--test_namespace', metavar='NAMES', nargs='+', default=['test'],
                    help='Namespaces of test analyses, default: test')
parser = basic_params(parser)
args = parser.parse_args()
path, outdir, namespace, seed = parse_arguments(args, args.name, model_path=True)


def plot_ref_alt(name):
    name = name.replace('_', '-')
    outputs_file = os.path.join(path, '{}_{}_outputs.npy'.format(namespace, name))
    outputs = np.load(outputs_file, allow_pickle=True)
    print('Loaded network outputs from {}'.format(outputs_file))
    labels_file = os.path.join(path, '{}_{}_labels.npy'.format(namespace, name))
    labels = list(np.load(labels_file, allow_pickle=True))
    print('Loaded sequences labels from {}'.format(labels_file))
    ids_file = os.path.join(path, '{}_{}.txt'.format(namespace, name))
    seq_ids = open(ids_file, 'r').read().strip().split('\n')
    print('Loaded sequences IDs from {}'.format(ids_file))
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
                    try:
                        id = l[-1]
                        pos = seq_ids.index(id)
                    except ValueError:
                        id = '{}:{}'.format(l[0].lstrip('chr'), l[1])
                        pos = seq_ids.index(id)
                    label_names[pos] = '{} {}'.format(l[3], l[4])
                    patients[pos] = id
                else:
                    if l[-1] == 'REF':
                        ref_seq = line.strip().upper()
                    seqs[pos] = line.strip().upper()
        num_snp = [len([a for a, r in zip(seq, ref_seq) if a != r]) for seq in seqs]
        min_nsnp = min([el for el in num_snp if el != 0])
        max_nsnp = max([el for el in num_snp if el != 0])
        num_snp = [(el - min_nsnp + 1) * 30 if el != 0 else 12 for el in num_snp]

    else:
        patients = seq_ids
        num_snp = {pat: 1 for pat in patients}
    print('Alternative and reference sequences read from {}'.format(seq_file))

    classes = get_classes_names(os.path.join(path, '{}_params.txt'.format(namespace)))
    xvalues = {'True class': [], 'False class': []}
    yvalues = {'True class': [], 'False class': []}
    sizes = {'True class': [], 'False class': []}

    correct_classified = 0
    for i, (label, n) in enumerate(zip(labels, label_names)):
        output = outputs[label]
        seq_pos = len([el for el in labels[:i] if el == label])
        xvalues['True class'].append(label * num_seqs + i + label + 1)
        correct_out = output[label][seq_pos]
        yvalues['True class'].append(correct_out)
        sizes['True class'].append(num_snp[i])
        classified = True
        for wrong_name in [el for el in classes if el != n]:
            wrong_label = classes.index(wrong_name)
            xvalues['False class'].append(wrong_label * num_seqs + i + wrong_label + 1)
            wrong_out = output[wrong_label][seq_pos]
            yvalues['False class'].append(wrong_out)
            if wrong_out >= correct_out:
                classified = False
            sizes['False class'].append(num_snp[i])
        if classified:
            correct_classified += 1
    print('Number of sequences: {}, number of classes: {}'.format(num_seqs, len(classes)))

    plt.figure(figsize=(20, 10))
    for legend_label, color, marker in zip(['True class', 'False class'], ['C2', 'C1'], ['*', 'o']):
        plt.scatter(xvalues[legend_label], yvalues[legend_label], s=sizes[legend_label], color=color, marker=marker,
                    label=legend_label, alpha=0.8)
    xticks = [la for el in xvalues.values() for la in el]
    xticks.sort()
    plt.xticks(xticks, patients*len(classes), fontsize=10, rotation=90, ha='center')
    plt.xlabel(('  ' * num_seqs).join(classes), fontsize=16)
    plt.ylabel('Output value', fontsize=16)
    plt.legend(fontsize=12, prop={'size': 16})
    plt.title('{} - {}'.format(namespace, name), fontsize=20)
    plt.ylim((0.45, 1.05))
    plt.text(0.0, 1.045, 'Correctly classified seqs: {}/{}\nNumber of SNPs in alt seqs: {}-{}'.
             format(correct_classified, num_seqs, min_nsnp, max_nsnp), fontsize=12, va='top')
    plt.tight_layout()
    plot_file = os.path.join(outdir, '{}_{}_ref:alt.png'.format(namespace, name))
    plt.savefig(plot_file)
    plt.show()
    print('Plot saved to {}'.format(plot_file))


for name in args.name:
    print('\nPlot for {}'.format(name))
    plot_ref_alt(name)
