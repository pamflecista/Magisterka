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
group1.add_argument('--valid', action='store_true',
                    help='Test model on the validation data of the given model')
group1.add_argument('--dataset', action='store', metavar='DATA', type=str, default=None,
                    help='Directory or file with the data for testing')
parser.add_argument('--batch_size', action='store', metavar='INT', type=int, default=64,
                    help='size of the batch, default: 64')
parser = basic_params(parser)
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, args.model, model_path=True)

batch_size = args.batch_size
if args.model is None:
    modelfile = os.path.join(path, '{}_last.model'.format(namespace))
elif args.model.startswith('/'):
    modelfile = args.model
else:
    modelfile = os.path.join(path, args.model)
if args.dataset is not None:
    data_name = args.dataset.split('/')[-1].split('.')[0].replace('_', '-')
    if os.path.isfile(args.dataset):
        data_dir = args.dataset
    elif os.path.isfile(os.path.join(path, args.dataset)):
        data_dir = os.path.join(path, args.dataset)
    else:
        print('Dataset {} not found'.format(args.dataset))
        raise ValueError
    subset = 'all:{}'.format(data_name)
else:
    data_dir = []
    subset = 'train' if args.train else 'valid' if args.valid else 'test'

network, data_dir, seq_len, ch, classes, _ = \
    params_from_file(os.path.join(output, '{}_params.txt'.format(namespace)), data_dir=data_dir)

# Define loggers for logfile and for results
(logger, results_table), old_results = build_loggers('test', output=output, namespace=namespace)

logger.info('\nTesting the network {} begins {}\nInput data: {} from {}\nOutput directory: {}\n'.format(
    modelfile, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), subset, data_dir, output))

# CUDA for PyTorch
use_cuda, device = check_cuda(logger)

# Build dataset for testing
t0 = time()
if subset.startswith('all'):
    names = []
else:
    names = open(os.path.join(output, '{}_{}.txt'.format(namespace, subset)), 'r').read().strip().split('\n')
dataset = SeqsDataset(data_dir, subset=names, seq_len=seq_len, name_pos=-1)
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
model.load_state_dict(torch.load(modelfile, map_location=torch.device(device)))
logger.info('\nModel from {} loaded in {:.2f} s'.format(modelfile, time() - t0))

test_loss_neurons = [[] for _ in range(num_classes)]
true, scores = [], []
confusion_matrix = np.zeros((num_classes, num_classes))
output_values = [[[] for _ in range(num_classes)] for _ in range(num_classes)]
logger.info('\n--- TESTING ---')
t0 = time()
with torch.no_grad():
    model.eval()
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
try:
    test_auc = calculate_auc(true, scores)
except ValueError:
    test_auc = [['-' for _ in dataset.classes] for _ in dataset.classes]

# Write the results
write_results(results_table, columns, ['test'], globals(), data_dir, subset)
np.save(os.path.join(output, '{}_{}_outputs'.format(namespace, subset)), np.array(output_values))
np.save(os.path.join(output, '{}_{}_labels'.format(namespace, subset)), np.array(true))
with open(os.path.join(output, '{}_{}.txt'.format(namespace, subset)), 'w') as f:
    f.write('\n'.join(dataset.IDs))

logger.info("Testing finished in {:.2f} min\nTest loss: {:1.3f}\n{:>35s}{:.5s}, {:.5s}, {:.5s}"
            .format((time() - t0) / 60, test_loss_reduced, '', 'SENSITIVITY', 'SPECIFICITY', 'AUC'))
logger.info("--{:>18s} :{:>5} seqs{:>22}".format('TESTING', len(dataset), "--"))
for cl, sens, spec, auc in zip(dataset.classes, test_sens, test_spec, test_auc):
    logger.info('{:>20} :{:>5} seqs - {:1.3f}, {:1.3f}, {:4s}'.format(cl, len(class_stage[cl]), sens, spec, auc[0]))
if isinstance(test_auc[0][0], float):
    logger.info(
            "--{:>18s} : {:1.3f}, {:1.3f}, {:1.3f}{:>12}".
            format('TESTING MEANS', *list(map(mean, [test_sens, test_spec, [el[0] for el in test_auc]])), "--"))
else:
    logger.info(
        "--{:>18s} : {:1.3f}, {:1.3f}, {}{:>12}".
            format('TESTING MEANS', *list(map(mean, [test_sens, test_spec])), test_auc[0][0], "--"))
logger.info('Testing of {} finished in {:.2f} min'.format(namespace, (time() - t0)/60))
