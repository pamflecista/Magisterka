from bin.datasets import SeqsDataset
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
from bin.common import *
from bin.networks import *
import math
import os
from statistics import mean
from time import time
from datetime import datetime
import numpy as np
import shutil
from collections import OrderedDict
import random

from bin.common import NET_TYPES

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
    'Network params': 'network_params',
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
    'AUC-neuron': ['aucINT', 'float-list']
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
parser.add_argument('data', action='store', metavar='DATASET', type=str, nargs='+',
                    help='Folder with the data for training and validation, if PATH is given, data is supposed to be ' +
                         'in PATH directory: [PATH]/data/[DATA]')
parser.add_argument('-n', '--network', action='store', metavar='NAME', type=str, default='basset',
                    help='type of the network to train, default: Basset Network')
parser = basic_params(parser)
parser.add_argument('--run', action='store', metavar='NUMBER', type=str, default='0',
                    help='number of the analysis, by default NAMESPACE is set to [NETWORK][RUN]')
parser.add_argument('--train', action='store', metavar='CHR', type=str, default='1-16',
                    help='chromosome(s) for training, if negative it means the number of chromosomes ' +
                         'which should be randomly chosen. Default: 1-16')
parser.add_argument('--val', action='store', metavar='CHR', type=str, default='17-20',
                    help='chromosome(s) for validation, if negative it means the number of chromosomes ' +
                         'which should be randomly chosen. Default: 17-20')
parser.add_argument('--test', action='store', metavar='CHR', type=str, default='21-23',
                    help='chromosome(s) for testing, if negative it means the number of chromosomes ' +
                         'which should be randomly chosen. Default: 21-23')
parser.add_argument('--optimizer', action='store', metavar='NAME', type=str, default='RMSprop',
                    help='optimization algorithm to use for training the network, default = RMSprop')
parser.add_argument('--loss_fn', action='store', metavar='NAME', type=str, default='CrossEntropyLoss',
                    help='loss function for training the network, default = CrossEntropyLoss')
parser.add_argument('--batch_size', action='store', metavar='INT', type=int, default=64,
                    help='size of the batch, default: 64')
parser.add_argument('--num_workers', action='store', metavar='INT', type=int, default=4,
                    help='how many subprocesses to use for data loading, default: 4')
parser.add_argument('--num_epochs', action='store', metavar='INT', type=int, default=300,
                    help='maximum number of epochs to run, default: 300')
parser.add_argument('--acc_threshold', action='store', metavar='FLOAT', type=float, default=0.9,
                    help='threshold of the validation accuracy - if gained training process stops, default: 0.9')
parser.add_argument('--no_adjust_lr', action='store_true',
                    help='no reduction of learning rate during training')
parser.add_argument('--seq_len', action='store', metavar='INT', type=int, default=2000,
                    help='Length of the input sequences to the network, default: 2000')
args = parser.parse_args()

batch_size, num_workers, num_epochs, acc_threshold, seq_len = args.batch_size, args.num_workers, args.num_epochs, \
                                                              args.acc_threshold, args.seq_len

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
network = NET_TYPES[network_name.lower()]
optim_method = OPTIMIZERS[optimizer_name]
lossfn = LOSS_FUNCTIONS[lossfn_name]
lr = 0.01
weight_decay = 0.0001
if args.no_adjust_lr:
    adjust_lr = False
else:
    adjust_lr = True

# Define files for logs and for results
(logger, results_table), old_results = build_loggers('train', output=output, namespace=namespace)

logger.info('\nAnalysis {} begins {}\nInput data: {}\nOutput directory: {}\n'.format(
    namespace, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), '; '.join(data_dir), output))

t0 = time()
train_chr, val_chr, test_chr = divide_chr(args.train, args.val, args.test)
if set(train_chr) & set(val_chr):
    logger.warning('WARNING - Chromosomes for training and validation overlap!')
elif set(train_chr) & set(test_chr):
    logger.warning('WARNING - Chromosomes for training and testing overlap!')
elif set(val_chr) & set(test_chr):
    logger.warning('WARNING - Chromosomes for validation and testing overlap!')

# CUDA for PyTorch
use_cuda, _ = check_cuda(logger)

dataset = SeqsDataset(data_dir, seq_len=seq_len)
num_classes = dataset.num_classes
classes = dataset.classes

# write header of results table
if not old_results:
    results_table, columns = results_header('train', results_table, RESULTS_COLS, classes)
else:
    columns = read_results_columns(results_table, RESULTS_COLS)

# Creating data indices for training, validation and test splits:
indices = dataset.get_chrs([train_chr, val_chr, test_chr])
class_stage = [dataset.get_classes(i) for i in indices]
train_indices, val_indices, test_indices = indices
train_len, val_len = len(train_indices), len(val_indices)
num_seqs = ' + '.join([str(len(el)) for el in [train_indices, val_indices, test_indices]])
for i, (n, ch, ind) in enumerate(zip(['train', 'valid', 'test'], map(make_chrstr, [train_chr, val_chr, test_chr]),
                                     [train_indices, val_indices, test_indices])):
    logger.info('\nChromosomes for {} ({}) - contain {} seqs:'.format(n, ch, len(indices[i])))
    for classname, el in class_stage[i].items():
        logger.info('{} - {}'.format(classname, len(el)))
    # Writing IDs for each split into file
    with open(os.path.join(output, '{}_{}.txt'.format(namespace, n)), 'w') as f:
        f.write('\n'.join([dataset.IDs[j] for j in ind]))

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

logger.info('\nTraining and validation datasets built in {:.2f} s'.format(time() - t0))

num_batches = math.ceil(train_len / batch_size)

model = network(dataset.seq_len)
network_params = model.params
optimizer = optim_method(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = lossfn()
best_acc = 0.0
# write parameters values into file
write_params(PARAMS, globals(), os.path.join(output, '{}_params.txt'.format(namespace)))
logger.info('\n--- TRAINING ---')
t = time()
for epoch in range(num_epochs):
    t0 = time()
    confusion_matrix = np.zeros((num_classes, num_classes))
    train_loss_neurons = [[] for _ in range(num_classes)]
    train_loss_reduced = 0.0
    true, scores = [], []
    if epoch == num_epochs - 1:
        train_output_values = [[[] for _ in range(num_classes)] for _ in range(num_classes)]
        val_output_values = [[[] for _ in range(num_classes)] for _ in range(num_classes)]
    for i, (seqs, labels) in enumerate(train_loader):
        model.train()
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

        with torch.no_grad():
            model.eval()
            outputs = model(seqs)
            losses = []
            for o, l in zip(outputs, labels):
                loss = -math.log((math.exp(o[l]))/(sum([math.exp(el) for el in o])))
                train_loss_neurons[l].append(loss)
                losses.append(loss)
            train_loss_reduced += mean(losses)

            _, indices = torch.max(outputs, axis=1)
            for ind, label in zip(indices, labels.cpu()):
                confusion_matrix[ind][label] += 1
                if epoch == num_epochs - 1:
                    train_output_values[label] = [el + [outp[j].cpu().item()] for j, el in enumerate(train_output_values[label])]

            true += labels.tolist()
            scores += outputs.tolist()

        if i % 10 == 0:
            logger.info('Epoch {}, batch {}/{}'.format(epoch+1, i, num_batches))

    # Call the learning rate adjustment function
    if not args.no_adjust_lr:
        adjust_learning_rate(epoch, optimizer)

    # Calculate metrics
    train_losses, train_sens, train_spec = calculate_metrics(confusion_matrix, train_loss_neurons)
    train_loss_reduced = train_loss_reduced / num_batches
    assert math.floor(mean([el for el in train_losses if el])*10/10) == math.floor(float(train_loss_reduced)*10/10)
    train_auc = calculate_auc(true, scores)

    with torch.no_grad():
        model.eval()
        confusion_matrix = np.zeros((num_classes, num_classes))
        val_loss_neurons = [[] for _ in range(num_classes)]
        true, scores = [], []
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
                    val_output_values[label] = [el + [outp[j].cpu().item()] for j, el in enumerate(val_output_values[label])]

            true += labels.tolist()
            scores += outputs.tolist()

    # Calculate metrics
    val_losses, val_sens, val_spec = calculate_metrics(confusion_matrix, val_loss_neurons)
    val_auc = calculate_auc(true, scores)

    # Save the model if the test acc is greater than our current best
    if mean(val_sens) > best_acc and epoch < num_epochs - 1:
        torch.save(model.state_dict(), os.path.join(output, "{}_{}.model".format(namespace, epoch + 1)))
        best_acc = mean(val_sens)

    # If it is a last epoch write neurons' outputs, labels and model
    if epoch == num_epochs - 1:
        logger.info('Last epoch - writing neurons outputs for each class!')
        np.save(os.path.join(output, '{}_train_outputs'.format(namespace)), np.array(train_output_values))
        np.save(os.path.join(output, '{}_valid_outputs'.format(namespace)), np.array(val_output_values))
        torch.save(model.state_dict(), os.path.join(output, '{}_last.model'.format(namespace)))

    # Write the results
    write_results(results_table, columns, ['train', 'val'], globals(), epoch+1)
    # Print the metrics
    logger.info("Epoch {} finished in {:.2f} min\nTrain loss: {:1.3f}\n{:>35s}{:.5s}, {:.5s}, {:.5s}"
                .format(epoch+1, (time() - t0)/60, train_loss_reduced, '', 'SENSITIVITY', 'SPECIFICITY', 'AUC'))
    logger.info("--{:>18s} :{:>5} seqs{:>22}".format('TRAINING', train_len, "--"))
    for cl, sens, spec, auc in zip(dataset.classes, train_sens, train_spec, train_auc):
        logger.info('{:>20} :{:>5} seqs - {:1.3f}, {:1.3f}, {:1.3f}'.format(cl, len(class_stage[0][cl]), sens, spec, auc[0]))
    logger.info("--{:>18s} :{:>5} seqs{:>22}".format('VALIDATION', val_len, "--"))
    for cl, sens, spec, auc in zip(dataset.classes, val_sens, val_spec, val_auc):
        logger.info('{:>20} :{:>5} seqs - {:1.3f}, {:1.3f}, {:1.3f}'.format(cl, len(class_stage[1][cl]), sens, spec, auc[0]))
    logger.info(
        "--{:>18s} : {:1.3f}, {:1.3f}, {:1.3f}{:>12}".
        format('TRAINING MEANS', *list(map(mean, [train_sens, train_spec, [el[0] for el in train_auc]])), "--"))
    logger.info(
        "--{:>18s} : {:1.3f}, {:1.3f}, {:1.3f}{:>12}\n\n".
        format('VALIDATION MEANS', *list(map(mean, [val_sens, val_spec, [el[0] for el in val_auc]])), "--"))

    if mean(val_sens) >= acc_threshold:
        logger.info('Validation accuracy threshold reached!')
        break

logger.info('Training for {} finished in {:.2f} min'.format(namespace, (time() - t)/60))
