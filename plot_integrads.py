import matplotlib.pyplot as plt
import argparse
from bin.common import *

COLORS = ['C{}'.format(i) for i in range(10)]

parser = argparse.ArgumentParser(description='Plot integrated gradients for given namespace '
                                             '(before it run calculate_integrads in order to create required files '
                                             'with gradients).')
parser = basic_params(parser)
parser.add_argument('--all_classes', action='store_true',
                    help='Plot outputs for all neurons (by default only output for the real label is showed)')
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, None)

param_file = os.path.join(path, 'integrads_{}_params.txt'.format(namespace))
with open(param_file) as f:
    for line in f:
        if line.startswith('Model file'):
            _, analysis_name = os.path.split(line.split(': ')[1].strip())
            analysis_name = analysis_name.split('_')[0]
        elif line.startswith('Seq file'):
            seq_file = line.split(': ')[1].strip()
        elif line.startswith('Seq IDs'):
            seq_ids = line.split(': ')[1].strip().split(', ')
        elif line.startswith('Seq length'):
            seq_len = int(line.split(': ')[1].strip())
        elif line.startswith('Seq labels'):
            seq_labels = list(map(int, line.split(': ')[1].strip().split(', ')))
        elif line.startswith('Classes'):
            classes = line.split(': ')[1].strip().split(', ')
        elif line.startswith('Number of trials'):
            trials = int(line.split(': ')[1].strip())
        elif line.startswith('Number of steps'):
            steps = int(line.split(': ')[1].strip())
results = {}
for name in classes:
    results[name] = np.load(os.path.join(path, 'integrads_{}_{}.npy'.format(namespace, '-'.join(name.split()))))

if args.all_classes:
    fig, axes = plt.subplots(nrows=len(seq_ids), ncols=len(classes), figsize=(12, 8), squeeze=False, sharex='col',
                             sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
    arg_classes = classes

else:
    fig, axes = plt.subplots(nrows=len(seq_ids), ncols=1, figsize=(12, 8), squeeze=False, sharex='col',
                             sharey='row', gridspec_kw={'hspace': 0.05, 'wspace': 0.05})
    arg_classes = ['True label']

leap = 50
min_value, max_value = 0, 0
for i, name in enumerate(arg_classes):
    for j, seq in enumerate(seq_ids):
        ax = axes[j, i]
        if j == 0:
            if args.all_classes:
                cc = COLORS[i]
            else:
                cc = 'black'
                for cl in classes:
                    axes[-1, -1].plot([], label=cl)
            ax.set_title(name, fontsize=15, color=cc)
        if i == 0:
            ax.set_ylabel(seq, fontsize=15, color=COLORS[seq_labels[j]])
        if args.all_classes:
            result = results[name][j]
        else:
            result = results[classes[seq_labels[j]]][j]
        result = [result[:, i:i+leap].flatten() for i in range(0, seq_len, leap)]
        ax.boxplot(result, showfliers=True, whis=15.0)
        labels = [i for i in np.arange(0, seq_len + 1,  seq_len//4)]
        xticks = [i for i in range(0, len(result)+1, len(result)//4)]
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels)
        min_value = [np.min(result) if np.min(result) < min_value else min_value][0]
        max_value = [np.max(result) if np.max(result) > max_value else max_value][0]
for ax in axes.flatten():
    ax.set_ylim((min_value, max_value))
if not args.all_classes:
    axes[-1, -1].legend(bbox_to_anchor=(0, -0.4), loc="lower left", ncol=4)

fig.suptitle('Integrated gradients - {}; {}; trials {}; steps {};'.format(*namespace.split('_')[:2], trials, steps),
             fontsize=15)
plt.tight_layout()
plt.show()
fig.savefig(os.path.join(output, namespace + 'integrads_{}.png'.format(namespace)))
