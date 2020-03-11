import argparse
from bin.common import *
from bin.datasets import SeqsDataset
import torch
from time import time
from bin.integrated_gradients import integrated_gradients

parser = argparse.ArgumentParser(description='Calculate and plot integrated gradients based on given sequences and '
                                             'network')
parser.add_argument('--model', action='store', metavar='NAME', type=str, default=None,
                    help='File with the model to check, if PATH is given, model is supposed to be in PATH directory, '
                         'if NAMESPACE is given model is supposed to be in [PATH]/results/[NAMESPACE]/ directory')
parser.add_argument('--seq', action='store', metavar='DATA', type=str, required=True,
                    help='File or folder with sequences to check, if PATH is given, file is supposed to be in '
                         '[PATH]/data/integrads/ directory.')
parser.add_argument('--trials', action='store', metavar='NUM', type=int, default=10,
                    help='Number of trials for calculating integrated gradients, default = 10.')
parser.add_argument('--steps', action='store', metavar='NUM', type=int, default=50,
                    help='Number of steps for each trial, default = 50.')
parser = basic_params(parser, plotting=True)
args = parser.parse_args()

path, output, namespace, seed = parse_arguments(args, args.model)

if args.model is None:
    model_file = os.path.join(path, 'results/{}/{}_last.model'.format(namespace, namespace))
else:
    model_file = args.model

if args.param is None:
    param_file = os.path.join(path, 'results/{}/{}_params.txt'.format(namespace, namespace))
else:
    param_file = args.param

seq_file = args.seq
_, seq_name = os.path.split(seq_file)
seq_name, _ = os.path.splitext(seq_name)

# CUDA for PyTorch
use_cuda, device = check_cuda(None)

network, _, seq_len, _, classes, analysis_name = params_from_file(param_file)

dataset = SeqsDataset(seq_file, seq_len=seq_len)
seq_ids = dataset.IDs
X, y = dataset.__getitem__(0)
labels = [y]
X = [X]
for i in range(1, len(dataset)):
    xx, yy = dataset.__getitem__(i)
    X.append(xx)
    labels.append(yy)
X = torch.stack(X, dim=0)

t0 = time()
# Build network
model = network(seq_len)
# Load weights from the file
model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
print('Model from {} loaded in {:.2f} s'.format(model_file, time() - t0))

results = {}
leap = 100
t0 = time()
for i, name in enumerate(classes):
    print('Calculating integrated gradients for {}'.format(name))
    r = np.squeeze(integrated_gradients(model, X, i, use_cuda=use_cuda, num_trials=args.trials, steps=args.steps), axis=1)
    np.save(os.path.join(output, 'integrads_{}_{}_{}'.format(analysis_name, seq_name, name)), r)
    results[name] = r
print('Gradients calculated in {:.2f} min and saved into {} directory'.format((time() - t0) / 60, output))
