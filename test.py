import argparse
from bin.common import *
from bin.datasets import *
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime
from time import time
from statistics import mean

from bin.common import NET_TYPES


RESULTS_COLS = OrderedDict({
    'Loss': ['losses', 'float-list'],
    'Sensitivity': ['sens', 'float-list'],
    'Specificity': ['spec', 'float-list'],
    'AUC-neuron': ['aucINT', 'float-list']
})

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
group1.add_argument('--dataset', action='store', metavar='DATA', type=str, nargs='+', default=[],
                    help='Directory with the data for testing')
parser.add_argument('--batch_size', action='store', metavar='INT', type=int, default=64,
                    help='size of the batch, default: 64')
parser = basic_params(parser)
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, args.model)
batch_size = args.batch_size

if args.model is None:
    modelfile = os.path.join(path, '{}_last.model'.format(namespace))
elif args.model.startswith('/'):
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
            data_dir = [line.split(':')[-1].strip()]
            if not data_dir:
                l = f.readline()
                while l.startswith('\t'):
                    data_dir.append(l.strip())
                    l = f.readline()
        elif line.strip().startswith('Input sequence length'):
            seq_len = int(line.split(':')[-1].strip())
        elif line.startswith('{} chr'.format(subset)):
            ch = read_chrstr(line.split(':')[-1].strip())

# Define loggers for logfile and for results
(logger, results_table), old_results = build_loggers('test', output=output, namespace=namespace)

logger.info('\nTesting the network {} begins {}\nInput data: {} from {}\nOutput directory: {}\n'.format(
    modelfile, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), subset, '; '.join(data_dir), output))

# CUDA for PyTorch
use_cuda, device = check_cuda(logger)

# Build dataset for testing
t0 = time()
if subset == 'all':
    names = []
else:
    names = open(os.path.join(output, '{}_{}.txt'.format(namespace, subset)), 'r').read().strip().split('\n')
dataset = SeqsDataset(data_dir, subset=names, seq_len=seq_len)
classes = dataset.classes
num_classes = len(classes)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
logger.info('\nTesting data contains {} seqs:'.format(len(dataset)))
class_stage = dataset.get_classes()
for classname, el in class_stage.items():
    logger.info('{} - {}'.format(classname, len(el)))

# Write header of results table
if not old_results:
    results_table, columns = results_header('test', results_table, RESULTS_COLS, classes)
else:
    columns = read_results_columns(results_table, RESULTS_COLS)
logger.info('\nTesting dataset built in {:.2f} s'.format(time() - t0))

num_batches = math.ceil(len(dataset) / batch_size)

t0 = time()
# Build network - this type which was used during training the model
model = network(seq_len)
# Load weights from the file
model.load_state_dict(torch.load(modelfile), map_location=torch.device(device))
logger.info('\nModel from {} loaded in {:.2f} s'.format(modelfile, time() - t0))

model.eval()
test_loss_neurons = [[] for _ in range(num_classes)]
true, scores = [], []
confusion_matrix = np.zeros((num_classes, num_classes))
output_values = [[[] for _ in range(num_classes)] for _ in range(num_classes)]
logger.info('\n--- TESTING ---')
t0 = time()
for i, (seqs, labels) in enumerate(loader):
    if use_cuda:
        seqs = seqs.cuda()
        labels = labels.cuda()
    seqs = seqs.float()
    labels = labels.long()

    outputs = model(seqs)

    for o, l in zip(outputs, labels):
        test_loss_neurons[l].append(-math.log((math.exp(o[l])) / (sum([math.exp(el) for el in o]))))

    _, indices = torch.max(outputs, axis=1)
    for ind, label, outp in zip(indices, labels.cpu(), outputs):
        confusion_matrix[ind][label] += 1
        output_values[label] = [el + [outp[j].cpu()] for j, el in enumerate(output_values[label])]

    true += labels.tolist()
    scores += outputs.tolist()

    if i % 10 == 0:
        logger.info('Batch {}/{}'.format(i, num_batches))

# Calculate metrics
test_losses, test_sens, test_spec = calculate_metrics(confusion_matrix, test_loss_neurons)
test_loss_reduced = math.floor(mean([el for el in test_losses if el])*10/10)
test_auc = calculate_auc(true, scores, num_classes)

# Write the results
write_results(results_table, columns, ['test'], globals(), data_dir, subset)

logger.info("Testing finished in {:.2f} min\nTest loss: {:1.3f}\n{:>35s}{:.5s}, {:.5s}, {:.5s}"
            .format((time() - t0) / 60, test_loss_reduced, '', 'SENSITIVITY', 'SPECIFICITY', 'AUC'))
logger.info("--{:>18s} :{:>5} seqs{:>22}".format('TESTING', len(dataset), "--"))
for cl, sens, spec, auc in zip(dataset.classes, test_sens, test_spec, test_auc):
    logger.info('{:>20} :{:>5} seqs - {:1.3f}, {:1.3f}, {:1.3f}'.format(cl, len(class_stage[cl]), sens, spec, auc[0]))
logger.info(
        "--{:>18s} : {:1.3f}, {:1.3f}{:>19}".
        format('TESTING MEANS', *list(map(mean, [test_sens, test_spec])), "--"))
logger.info('Testing of {} finished in {:.2f} min'.format(namespace, (time() - t0)/60))
