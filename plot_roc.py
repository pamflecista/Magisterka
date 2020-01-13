import argparse
from bin.common import *
from collections import Counter

parser = argparse.ArgumentParser(description='Plot ROC curve for the given data')
parser.add_argument()
args = parser.parse_args()

infile = os.path.join(output, '{}_{}_outputs.npy'.format(namespace, subset))
if not os.path.isfile(infile):
    print('Output for this subset has not been written yet. Please run testing on this data subset to create {} file'.format(infile))
    sys.exit()

outputs = np.load(infile)
for i in range(outputs.shape[0]):
    labels = []
    for label, row in enumerate(outputs):
        labels += [label for _ in row[i]]
    values = outputs[:, i].values
    print(Counter(values))
    plot_roc(labels, values, name='ROC for {i} class')

