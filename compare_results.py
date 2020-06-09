import matplotlib.pyplot as plt
import argparse
from bin.common import *
from statistics import median


parser = argparse.ArgumentParser(
    description='Compare validation results from different networks - make scatter plot')
parser = basic_params(parser)
parser.add_argument('-n', '--network', action='store', metavar='COL', nargs='+', type=str,
                    default=['basset30', 'custom40'],
                    help='Names of networks to compare')
parser.add_argument('-c', '--column', action='store', metavar='NAME', type=str, default='auc',
                    help='Measure on which plot should be based, one of: loss, sens, spec, auc. Default auc.')
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, None)
column = args.column

if 'results' not in path:
    path = os.path.join(path, 'results')

classes = []
epochs = []
networks = args.network
print('Scatter plot for comparing networks: {}'.format(', '.join(networks)))

for n in networks:
    param_file = os.path.join(path, n, '{}_params.txt'.format(n))
    _, _, _, _, cl, _, num_epochs = params_from_file(param_file)
    if not classes:
        classes = cl
    else:
        assert classes == cl
    epochs.append(num_epochs)
print('Found {} common classes: {}'.format(len(classes), ', '.join(classes)))


values = []
for name, num_epochs in zip(networks, epochs):
    table = os.path.join(path, name, '{}_train_results.tsv'.format(name))
    with open(table, 'r') as f:
        header = f.readline().strip().split('\t')
        try:
            colnum = header.index(SHORTCUTS[column])
        except ValueError:
            colnum = [i for i, el in enumerate(header) if SHORTCUTS[column] in el]
        v = []
        for e, line in enumerate(f):
            line = line.strip().split('\t')
            if int(line[0]) == num_epochs and line[1] == 'val':
                if isinstance(colnum, list):
                    vv = []
                    for c, clname in zip(colnum, classes):
                        vv.append([float(el) if el != '-' else 0.0 for el in line[c].split(', ')][classes.index(clname)])
                    v.append(vv)
                else:
                    v.append([float(el) if el != '-' else np.nan for el in line[colnum].split(', ')])
        values.append(v)

fig = plt.figure(figsize=(10, 15))
xvalues = [i+1 for i in range(len(classes))]
for i, (name, v) in enumerate(zip(networks, values)):
    plt.scatter(xvalues, v, label=name)
plt.xticks(xvalues, classes, fontsize=15)
plt.legend()
plt.title('Comparison of results', fontsize=20)
plt.ylabel(SHORTCUTS[column], fontsize=15)
plt.show()
