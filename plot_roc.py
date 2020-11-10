import argparse
from bin.common import *
from collections import Counter
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

COLORS = ['C{}'.format(i) for i in range(10)]

parser = argparse.ArgumentParser(description='Plot ROC curve for the given data')
parser = basic_params(parser)
parser.add_argument('--subset', action='store', metavar='NAME', type=str, default='test',
                    help='Subset of data based on which plot ROC curve. Options: train, '
                    'test, valid, or all:[NAME] where NAME is a dataset name. Default: test')
parser.add_argument('--outputs_file', action='store', metavar='FILE', type=str, default=None,
                    help='Name of the file with outputs based on which ROC curve should be plotted, '
                         'default: [PATH]/[NAMESPACE]_[SUBSET]_outputs.npy')
parser.add_argument('--labels_file', action='store', metavar='FILE', type=str, default=None,
                    help='Name of the file with labels based on which ROC curve should be plotted, '
                         'default: [PATH]/[NAMESPACE]_[SUBSET]_labels.npy')
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, args.outputs_file, model_path=True)
subset = args.subset

if args.outputs_file is None:
    infile = os.path.join(output, '{}_{}_outputs.npy'.format(namespace, subset))
else:
    infile = args.outputs_file
if args.labels_file is None:
    labelfile = os.path.join(output, '{}_{}_labels.npy'.format(namespace, subset))
else:
    labelfile = args.labels_file
assert os.path.isfile(infile), 'Output for this subset has not been written yet. ' \
                               'Please run testing based on it in order to create {} file'.format(infile)
outputs = np.load(infile, allow_pickle=True)
# labels = np.load(labelfile, allow_pickle=True)
neurons = get_classes_names(os.path.join(path, '{}_params.txt'.format(namespace)))
assert outputs.shape[0] == outputs.shape[1] == len(neurons)
num_classes = len(neurons)
fig, axes = plt.subplots(nrows=1, ncols=outputs.shape[0], figsize=(15, 8))
for (i, ax), neuron in zip(enumerate(axes), neurons):
    labels = []
    for label, row in enumerate(outputs):
        labels += [label for _ in row[i]]
    y_true = [1 if el == i else 0 for el in labels]
    values = []
    values = [item for sublist in [values + el for el in outputs[:, i].flatten()] for item in sublist]
    if len(set(values)) < len(values):
        print('Repetitions for {}'.format(neuron))
        for value, counts in {k: v for k, v in Counter(values).items() if v > 1}.items():
            ll = Counter([el for el, v in zip(labels, values) if v == value])
            print('value {}: repeated {} times, labels: {}'.format(value, counts, ll))
    fpr, tpr, thresholds = roc_curve(y_true, values)
    ax.plot(fpr, tpr, label=neuron, color=COLORS[i])
    for neg in [j for j in range(num_classes) if j != i]:
        y_help = [1 if el == i else (0 if el == neg else -1) for el in labels]
        y_score = [el for use, el in zip(y_help, values) if use != -1]
        y_true = [el for el in y_help if el != -1]
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=neurons[neg], color=COLORS[neg])
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_title(neuron)
    if i == 0:
        ax.set_ylabel('True positive rate')
        ax.legend(bbox_to_anchor=(0, -0.07), loc="upper left", ncol=4)
    ax.set_xlabel("False positive rate")

fig.suptitle('{} - {}'.format(namespace, subset))
plt.show()
fig.savefig(os.path.join(output, '{}_{}_roc.png'.format(namespace, subset)))
