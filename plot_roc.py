import argparse
from bin.common import *
from collections import Counter
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot ROC curve for the given data')
parser = basic_params(parser)
parser.add_argument('--subset', action='store', metavar='NAME', type=str, default='test',
                    help='Subset of data based on which plot ROC curve. Options: train, '
                    'test, valid, or all:[NAME] where NAME is a dataset name. Default: test')
parser.add_argument('--infile', action='store', metavar='FILE', type=str, default=None,
                    help='Name of the file with outputs based on which ROC curve should be plotted, '
                         'default: [PATH]/[NAMESPACE]_[SUBSET]_outputs.npy')
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, args.infile)
subset = args.subset

if args.infile is None:
    infile = os.path.join(output, '{}_{}_outputs.npy'.format(namespace, subset))
else:
    infile = args.infile
assert os.path.isfile(infile), 'Output for this subset has not been written yet. ' \
                               'Please run testing based on it in order to create {} file'.format(infile)
outputs = np.load(infile, allow_pickle=True)
neurons = get_classes_names(os.path.join(path, '{}_params.txt'.format(namespace)))
assert outputs.shape[0] == outputs.shape[1] == len(neurons)
num_classes = len(neurons)
fig, axes = plt.subplots(nrows=1, ncols=outputs.shape[0], figsize=(15, 8))
for (i, ax), neuron in zip(enumerate(axes), neurons):
    labels = []
    for label, row in enumerate(outputs):
        labels += [label for _ in row[i]]
    values = outputs[:, i]
    print(Counter(values))
    fpr, tpr, thresholds = roc_curve(labels, values)
    ax.plot(fpr, tpr)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_title(neuron)
    if i == 0:
        ax.set_ylabel('True positive rate')
    ax.set_xlabel("False positive rate")

fig.suptitle('ROC curves for {} results from {}'.format(subset, namespace))
plt.show()
fig.savefig(os.path.join(output, '{}_{}_roc.png'.format(namespace, subset)))
