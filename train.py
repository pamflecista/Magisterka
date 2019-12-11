from bin.datasets import SeqsDataset
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
from bin.funcs import *
from bin.networks import BassetNetwork
import math
import os
from statistics import mean
import logging
from time import time
from datetime import datetime
import numpy as np
import shutil
from sklearn.metrics import roc_auc_score
from collections import OrderedDict


NET_TYPES = {
    'Basset': BassetNetwork
}

OPTIMIZERS = {
    'RMSprop': optim.RMSprop,
    'Adam': optim.Adam
}

LOSS_FUNCTIONS = {
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'MSELoss': nn.MSELoss
}

PARAMS = OrderedDict({
    'Name of the analysis': 'namespace',
    'Network type': 'network_name',
    'Possible classes': 'classes',
    'Number of epochs': 'num_epochs',
    'Number of seqs': 'num_seqs',
    'Batch size': 'batch_size',
    'Training chr': 'train_chr',
    'Validation chr': 'val_chr',
    'Test chr': 'test_chr',
    'Data directory': 'data_dir',
    'Random seed': 'seed',
    'CUDA available': 'use_cuda',
    'Optimizer': 'optimizer_name',
    'Loss function': 'lossfn_name',
    'Learning rate': 'lr',
    'Adjusting lr': 'adjust_lr',
    'Weight decay': 'weight_decay'
})

RESULTS_COLS = OrderedDict({
    'Loss': ['losses', 'float-list'],
    'Sensitivity': ['sens', 'float-list'],
    'Specificity': ['spec', 'float-list'],
    'AUC': ['auc', 'float']
})


def adjust_learning_rate(epoch, optimizer):
    lr = 0.001

    if epoch > 500:
        lr = lr / 100000
    elif epoch > 400:
        lr = lr / 10000
    elif epoch > 300:
        lr = lr / 1000
    elif epoch > 200:
        lr = lr / 100
    elif epoch > 100:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


parser = argparse.ArgumentParser(description='Train network based on given data')
parser.add_argument('data', action='store', metavar='DIR', type=str, nargs='+',
                    help='Folder with the data for training and validation, if PATH is given, data is supposed to be ' +
                         'in PATH directory: [PATH]/data/[DATA]')
parser.add_argument('-n', '--network', action='store', metavar='NAME', type=str, default='Basset',
                    help='Type of the network to train, default: Basset Network')
parser = basic_params(parser)
parser.add_argument('--run', action='store', metavar='NUMBER', type=str, default='0',
                    help='Number of the analysis, by default NAMESPACE is set to [NETWORK][RUN]')
parser.add_argument('--train', action='store', metavar='CHR', type=str, default='1-16',
                    help='Chromosome(s) for training, if negative it means the number of chromosomes ' +
                         'which should be randomly chosen. Default: 1-16')
parser.add_argument('--val', action='store', metavar='CHR', type=str, default='17-20',
                    help='Chromosome(s) for validation, if negative it means the number of chromosomes ' +
                         'which should be randomly chosen. Default: 17-20')
parser.add_argument('--test', action='store', metavar='CHR', type=str, default='21-23',
                    help='Chromosome(s) for testing, if negative it means the number of chromosomes ' +
                         'which should be randomly chosen. Default: 21-23')
parser.add_argument('--optimizer', action='store', metavar='NAME', type=str, default='RMSprop',
                    help='Optimization algorithm to use for training the network, default = RMSprop')
parser.add_argument('--loss_fn', action='store', metavar='NAME', type=str, default='CrossEntropyLoss',
                    help='Loss function for training the network, default = CrossEntropyLoss')
parser.add_argument('--batch_size', action='store', metavar='INT', type=int, default=64,
                    help='Size of the batch, default: 64')
parser.add_argument('--num_workers', action='store', metavar='INT', type=int, default=4,
                    help='How many subprocesses to use for data loading, default: 4')
parser.add_argument('--num_epochs', action='store', metavar='INT', type=int, default=300,
                    help='Maximum number of epochs to run, default: 300')
parser.add_argument('--acc_threshold', action='store', metavar='FLOAT', type=float, default=0.9,
                    help='Threshold of the validation accuracy - if gained training process stops, default: 0.9')
parser.add_argument('--no_adjust_lr', action='store_true',
                    help='No reduction of learning rate during training')
args = parser.parse_args()

batch_size, num_workers, num_epochs, acc_threshold = args.batch_size, args.num_workers, args.num_epochs, \
                                                     args.acc_threshold

path, output, namespace, seed = parse_arguments(args, args.data, namesp=args.network + args.run)
# create folder for the output files
if os.path.isdir(output):
    shutil.rmtree(output)
try:
    os.mkdir(output)
except FileNotFoundError:
    os.mkdir(os.path.join(path, 'results'))
    os.mkdir(output)
# establish data directories
if args.path is not None:
    data_dir = [os.path.join(path, 'data', d) for d in args.data]
else:
    data_dir = args.data
    if os.path.isdir(data_dir[0]):
        path = data_dir[0]
# set the random seed
torch.manual_seed(seed)
np.random.seed(seed)
# set other params
network_name = args.network
optimizer_name = args.optimizer
lossfn_name = args.loss_fn
network = NET_TYPES[network_name]
optim_method = OPTIMIZERS[optimizer_name]
lossfn = LOSS_FUNCTIONS[lossfn_name]
lr = 0.01
weight_decay = 0.0001
if args.no_adjust_lr:
    adjust_lr = False
else:
    adjust_lr = True

# Define files for logs and for results
formatter = logging.Formatter('%(message)s')
logger = logging.getLogger('verbose')
results_table = logging.getLogger('results')
cmd_handler = logging.StreamHandler()
log_handler = logging.FileHandler(os.path.join(output, '{}.log'.format(namespace)))
results_handler = logging.FileHandler(os.path.join(output, '{}_results.tsv'.format(namespace)))
for logg, handlers in zip([logger, results_table], [[cmd_handler, log_handler], [results_handler]]):
    for handler in handlers:
        handler.setFormatter(formatter)
        logg.addHandler(handler)
    logg.setLevel(logging.INFO)

logger.info('\nAnalysis {} begins {}\nInput data: {}\nOutput directory: {}\n'.format(
    namespace, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), '; '.join(data_dir), output))
results_table.info('Epoch\tStage\t{}'.format('\t'.join(RESULTS_COLS.keys())))

t0 = time()
train_chr = read_chrstr(args.train)
val_chr = read_chrstr(args.val)
test_chr = read_chrstr(args.test)
if set(train_chr) & set(val_chr):
    logger.warning('WARNING - Chromosomes for training and validation overlap!')
elif set(train_chr) & set(test_chr):
    logger.warning('WARNING - Chromosomes for training and testing overlap!')
elif set(val_chr) & set(test_chr):
    logger.warning('WARNING - Chromosomes for validation and testing overlap!')

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    logger.info('--- CUDA available ---')
else:
    logger.info('--- CUDA not available ---')

dataset = SeqsDataset(data_dir)
num_classes = dataset.num_classes
classes = dataset.classes

# Creating data indices for training, validation and test splits:
indices, data_labels = dataset.get_chrs([train_chr, val_chr, test_chr])
train_indices, val_indices, test_indices = indices
train_len, val_len = len(train_indices), len(val_indices)
num_seqs = ' + '.join([str(len(el)) for el in [train_indices, val_indices, test_indices]])
for i, (n, ch, ind) in enumerate(zip(['train', 'valid', 'test'], [args.train, args.val, args.test],
                                     [train_indices, val_indices, test_indices])):
    logger.info('\nChromosomes for {} ({}) - contain {} seqs:'.format(n, ch, len(indices[i])))
    for j, cl in enumerate(dataset.classes):
        logger.info('{} - {}'.format(data_labels[i][j], cl))
    # Writing IDs for each split into file
    with open(os.path.join(output, '{}_{}.txt'.format(namespace, n)), 'w') as f:
        f.write('\n'.join([dataset.IDs[j] for j in ind]))

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

logger.info('\nTraining, validation and testing datasets built in {:.2f} s'.format(time() - t0))

num_batches = math.ceil(train_len / batch_size)

model = network(dataset.seq_len)
optimizer = optim_method(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = lossfn()
best_acc = 0.0
# write parameters values into file
write_params(PARAMS, globals(), os.path.join(output, '{}_params.txt'.format(namespace)))
logger.info('\n--- TRAINING ---')
t = time()
for epoch in range(num_epochs):
    stage = 'train'
    t0 = time()
    model.train()
    confusion_matrix = np.zeros((num_classes, num_classes))
    train_loss_neurons = [[] for _ in range(num_classes)]
    train_loss_reduced = 0.0
    y_true, y_score = [], []
    for i, (seqs, labels) in enumerate(train_loader):
        if use_cuda:
            seqs = seqs.cuda()
            labels = labels.cuda()
            model.cuda()
        seqs = seqs.float()
        labels = labels.long()

        optimizer.zero_grad()
        outputs = model(seqs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        for o, l in zip(outputs, labels):
            train_loss_neurons[l].append(-math.log((math.exp(o[l]))/(sum([math.exp(el) for el in o]))))
        train_loss_reduced += loss.cpu().data

        _, indices = torch.max(outputs, axis=1)
        for ind, label in zip(indices, labels.cpu()):
            confusion_matrix[ind][label] += 1

        y_true += labels.tolist()
        y_score += outputs.tolist()

        if i % 10 == 0:
            logger.info('Epoch {}, batch {}/{}'.format(epoch+1, i, num_batches))

    # Call the learning rate adjustment function
    if not args.no_adjust_lr:
        adjust_learning_rate(epoch, optimizer)

    # Calculate metrics
    train_losses, train_sens, train_spec = calculate_metrics(confusion_matrix, train_loss_neurons)
    train_loss_reduced = train_loss_reduced / num_batches
    assert math.floor(mean([el for el in train_losses if el])*10/10) == math.floor(float(train_loss_reduced)*10/10)
    y_score = [[round(v, 3) for v in el] for el in y_score]
    y_scaled = [[round(v / sum(el), 3) for v in el[:-1]] for el in y_score]
    y_scaled = [el + [round(1.0 - sum(el), 3)] for el in y_scaled]
    train_auc = roc_auc_score(y_true, y_scaled, multi_class='ovr')

    # Write results to the table
    '''results_table.info('{}\t{}\t{}\t{}\t{}\t{}'.format(
        epoch + 1,
        'train',
        ', '.join(['{:.2f}'.format(el) if el else '-' for el in train_losses]),
        ', '.join(['{:.2f}'.format(el) for el in train_sens]),
        ', '.join(['{:.2f}'.format(el) for el in train_spec]),
        ', '.join(['{:.2f}'.format(el) for el in train_auc])
    ))'''

    stage = 'val'
    with torch.set_grad_enabled(False):
        model.eval()
        confusion_matrix = np.zeros((num_classes, num_classes))
        val_loss_neurons = [[] for _ in range(num_classes)]
        if epoch == num_epochs - 1:
            output_values = [[[] for _ in range(num_classes)] for _ in range(num_classes)]
        y_true, y_score = [], []
        for i, (seqs, labels) in enumerate(val_loader):
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
                if epoch == num_epochs - 1:
                    output_values[label] = [el + [outp[j].cpu()] for j, el in enumerate(output_values[label])]

            y_true += labels.tolist()
            y_score += outputs.tolist()

    # If it is a last epoch write neurons' outputs
    if epoch == num_epochs - 1:
        logger.info('Last epoch - writing neurons outputs for each class!')
        np.save(os.path.join(output, '{}_outputs'.format(namespace)), np.array(output_values))
    # Calculate metrics
    val_losses, val_sens, val_spec = calculate_metrics(confusion_matrix, val_loss_neurons)
    y_score = [[round(v, 3) for v in el] for el in y_score]
    y_scaled = [[round(v / sum(el), 3) for v in el[:-1]] for el in y_score]
    y_scaled = [el + [round(1.0 - sum(el), 3)] for el in y_scaled]
    val_auc = roc_auc_score(y_true, y_scaled, multi_class='ovr')

    '''# Write results to the table
    results_table.info('{}\t{}\t{}\t{}\t{}\t{}'.format(
        epoch + 1,
        'valid',
        ', '.join(['{:.2f}'.format(el) if el else '-' for el in val_losses]),
        ', '.join(['{:.2f}'.format(el) for el in val_sens]),
        ', '.join(['{:.2f}'.format(el) for el in val_spec]),
        ', '.join(['{:.2f}'.format(el) for el in val_auc])
    ))'''

    # Save the model if the test acc is greater than our current best
    if mean(val_sens) > best_acc:
        torch.save(model.state_dict(), os.path.join(output, "{}_{}.model".format(namespace, epoch+1)))
        best_acc = mean(val_sens)

    # Write the results
    write_results(results_table, RESULTS_COLS.values(), globals(), epoch)
    # Print the metrics
    logger.info("Epoch {} finished in {:.2f} min\nTrain loss: {:1.3f}\n Train AUC: {:1.3f}\n{:>35s}{:.5s}, {:.5s}, {:.5s}"
                .format(epoch+1, (time() - t0)/60, train_loss_reduced, train_auc, '', 'SENSITIVITY', 'SPECIFICITY', 'AUC'))
    logger.info("--{:>18s} :{:>5} seqs{:>15}".format('TRAINING', train_len, "--"))
    for cl, seqs, sens, spec, in zip(dataset.classes, data_labels[0], train_sens, train_spec):
        logger.info('{:>20} :{:>5} seqs - {:1.3f}, {:1.3f}'.format(cl, seqs, sens, spec))
    logger.info("--{:>18s} :{:>5} seqs{:>15}".format('VALIDATION', val_len, "--"))
    logger.info("Validation AUC: {:1.3f}".format(val_auc))
    for cl, seqs, sens, spec in zip(dataset.classes, data_labels[1], val_sens, val_spec):
        logger.info('{:>20} :{:>5} seqs - {:1.3f}, {:1.3f}'.format(cl, seqs, sens, spec))
    logger.info(
        "--{:>18s} : {:1.3f}, {:1.3f}{:>12}".
        format('TRAINING MEANS', *list(map(mean, [train_sens, train_spec])), "--"))
    logger.info(
        "--{:>18s} : {:1.3f}, {:1.3f}{:>12}\n\n".
        format('VALIDATION MEANS', *list(map(mean, [val_sens, val_spec])), "--"))

    if mean(val_sens) >= acc_threshold:
        logger.info('Validation accuracy threshold reached!')
        break

logger.info('Training for {} finished in {:.2f} min'.format(namespace, (time() - t)/60))
