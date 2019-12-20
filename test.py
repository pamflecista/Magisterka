import argparse
from bin.common import *
from bin.datasets import *
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime

from bin.common import NET_TYPES

parser = argparse.ArgumentParser(description='Test given network based on the given data, '
                                             'by default test subset established during training of the model is used')
parser.add_argument('--model', action='store', metavar='NAME', type=str, default=None,
                    help='File with the model to test, if PATH is given, model is supposed to be in PATH directory, '
                         'if NAMESPACE is given model is supposed to be in [PATH]/results/[NAMESPACE]/ directory')
group1 = parser.add_mutually_exclusive_group(required=False)
group1.add_argument('--train', action='store_true',
                    help='Test model on the train data of the given model')
group1.add_argument('--val', action='store_true',
                    help='Test model on the validation data of the given model')
group1.add_argument('--dataset', action='store', metavar='DATA', type=str, nargs='+',
                    help='Directory with the data for testing')
parser = basic_params(parser)
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, args.model)

if args.model is None:
    modelfile = os.path.join(path, '{}_last.model'.format(namespace))
elif not args.model.startswith(path):
    modelfile = args.model
else:
    modelfile = os.path.join(path, args.model)

data_dir = args.dataset
allseqs = True if data_dir else False

subset = 'train' if args.train else 'val' if args.val else 'test' if not allseqs else 'all'
seq_len = 2000
with open(os.path.join(output, '{}_params.txt'.format(namespace)), 'r') as f:
    for line in f:
        if line.startswith('Network type'):
            network = NET_TYPES[line.split(':')[-1].strip().lower()]
        elif line.startswith('Data directory') and not data_dir:
            data_dir = line.split(':')[-1].strip()
        elif line.startswith('Batch size'):
            batch_size = int(line.split(':')[-1].strip())
        elif line.strip().startswith('Input sequence length'):
            seq_len = int(line.split(':')[-1].strip())
        elif line.startswith('{} chr'.format(subset)):
            ch = read_chrstr(line.split(':')[-1].strip())

# Define loggers for logfile and for results
logger, results_table = build_loggers('test', output=output, namespace=namespace)

logger.info('\nTesting the network {} begins {}\nInput data: {} from {}\nOutput directory: {}\n'.format(
    modelfile, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), subset, '; '.join(data_dir), output))

# CUDA for PyTorch
use_cuda = check_cuda(logger)

# Build dataset for testing
dataset = SeqsDataset(data_dir, seq_len=seq_len)
classes = dataset.classes
num_classes = len(classes)
if subset == 'all':
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    logger.info('\nTesting data contains {} seqs:'.format(len(dataset)))
    indices = None
else:
    indices = list(map(int, open(os.path.join(output, '{}_{}.txt'.format(namespace, subset)), 'r').read().strip().split('\n')))
    sampler = SubsetRandomSampler(indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    logger.info('\nChromosomes for testing ({}) contain {} seqs:'.format(make_chrstr(ch), len(indices)))
for classname, el in dataset.get_classes(indices).items():
    logger.info('{} - {}'.format(classname, len(el)))

# Write header of results table
results_table = test_results_header(results_table, classes)

# Build network - this type which was used during training the model
model = network(seq_len)
# Load weights from the file
model.load_state_dict(torch.load(modelfile))
model.eval()

val_loss_neurons = [[] for _ in range(num_classes)]
true, scores = [], []
confusion_matrix = np.zeros((num_classes, num_classes))
output_values = [[[] for _ in range(num_classes)] for _ in range(num_classes)]
for i, (seqs, labels) in enumerate(loader):
    if use_cuda:
        seqs = seqs.cuda()
        labels = labels.cuda()
    seqs = seqs.float()
    labels = labels.long()

    outputs = model(seqs)

    for o, l in zip(outputs, labels):
        val_loss_neurons[l].append(-math.log((math.exp(o[l])) / (sum([math.exp(el) for el in o]))))

    _, indices = torch.max(outputs, axis=1)
    for ind, label, outp in zip(indices, labels.cpu(), outputs):
        confusion_matrix[ind][label] += 1
        output_values[label] = [el + [outp[j].cpu()] for j, el in enumerate(output_values[label])]

    true += labels.tolist()
    scores += outputs.tolist()

# Calculate metrics
val_losses, val_sens, val_spec = calculate_metrics(confusion_matrix, val_loss_neurons)
val_auc = calculate_auc(true, scores, num_classes)

# Write the results
write_results(results_table, globals(), )
