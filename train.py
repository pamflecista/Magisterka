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
import shutil
import numpy as np
from itertools import product

NET_TYPES = {
    'Basset': BassetNetwork
}

OPTIMIZERS = {
    'RMSprop': optim.RMSprop,
    'Adam': optim.Adam
}


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
                    help='Folder with the data for training and validation, if PATH is given, data is supposed to be '
                         'in PATH directory: [PATH]/[DATA]')
parser.add_argument('-n', '--network', action='store', metavar='NAME', type=str, default='Basset',
                    help='Type of the network to train, default: Basset Network')
parser.add_argument('-train', action='store', metavar='CHR', type=str, default='1-13',
                    help='Chromosome(s) for training, if negative it means the number of chromosomes '
                         'which should be randomly chosen. Default: 1-13')
parser.add_argument('-val', action='store', metavar='CHR', type=str, default='14-18',
                    help='Chromosome(s) for validation, if negative it means the number of chromosomes '
                         'which should be randomly chosen. Default: 14-18')
parser.add_argument('-test', action='store', metavar='CHR', type=str, default='19-22',
                    help='Chromosome(s) for testing, if negative it means the number of chromosomes '
                         'which should be randomly chosen. Default: 19-22')
parser.add_argument('-o', '--output', action='store', metavar='DIR', type=str, default=None,
                    help='Output directory, default: [PATH]/results/[NAMESPACE]')
parser.add_argument('-p', '--path', action='store', metavar='DIR', type=str, default=None,
                    help='Working directory, default: ./')
parser.add_argument('--namespace', action='store', metavar='NAME', type=str, default=None,
                    help='Namespace of the analysis, default: [NETWORK]')
parser.add_argument('--run', action='store', metavar='NUMBER', type=str, default='0',
                    help='Number of the analysis, by default NAMESPACE is set to [NETWORK][RUN]')
parser.add_argument('--optimizer', action='store', metavar='NAME', type=str, default='RMSprop',
                    help='Optimization algorithm to use for training the network, default = RMSprop')
parser.add_argument('-b', '--batch_size', action='store', metavar='INT', type=int, default=64,
                    help='Size of the batch, default: 64')
parser.add_argument('--num_workers', action='store', metavar='INT', type=int, default=4,
                    help='How many subprocesses to use for data loading, default: 4')
parser.add_argument('--num_epochs', action='store', metavar='INT', type=int, default=500,
                    help='Maximum number of epochs to run, default: 500')
parser.add_argument('--acc_threshold', action='store', metavar='FLOAT', type=float, default=0.9,
                    help='Threshold of the validation accuracy - if gained training process stops, default: 0.9')
parser.add_argument('--no_adjust_lr', action='store_true',
                    help='No reduction of learning rate during training')
args = parser.parse_args()

batch_size, num_workers, num_epochs, acc_threshold = args.batch_size, args.num_workers, args.num_epochs, \
                                                     args.acc_threshold

network = NET_TYPES[args.network]
optim_method = OPTIMIZERS[args.optimizer]
if args.namespace is None:
    namespace = args.network + args.run
else:
    namespace = args.namespace

if args.path is not None:
    path = args.path
    data_dir = [os.path.join(path, d) for d in args.data]
else:
    data_dir = args.data
    if os.path.isdir(data_dir[0]):
        path = data_dir[0]
    else:
        path = '/' + '/'.join(data_dir[0].split('/')[:-1])

if args.output is not None:
    output = args.output
else:
    output = os.path.join(path, 'results', namespace)
    if os.path.isdir(output):
        shutil.rmtree(output)
    os.mkdir(output)


handlers = [logging.FileHandler(os.path.join(output, '{}.log'.format(namespace))),
            logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)

logging.info('Analysis {} begins!\nInput data: {}\nOutput directory: {}\n'.format(namespace, '; '.join(data_dir), output))

t0 = time()
train_chr = read_chrstr(args.train)
val_chr = read_chrstr(args.val)
test_chr = read_chrstr(args.test)
if set(train_chr) & set(val_chr):
    logging.warning('WARNING - Chromosomes for training and validation overlap!')
elif set(train_chr) & set(test_chr):
    logging.warning('WARNING - Chromosomes for training and testing overlap!')
elif set(val_chr) & set(test_chr):
    logging.warning('WARNING - Chromosomes for validation and testing overlap!')

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    logging.info('--- CUDA available ---')
else:
    logging.info('--- CUDA not available ---')

dataset = SeqsDataset(data_dir)

num_classes = dataset.num_classes

# Creating data indices for training, validation and test splits:
indices, data_labels, seq_len = dataset.get_chrs([train_chr, val_chr, test_chr])
train_indices, val_indices, test_indices = indices
train_len, val_len = len(train_indices), len(val_indices)
for i, (n, c, ind) in enumerate(zip(['train', 'valid', 'test'], [args.train, args.val, args.test],
                                    [train_indices, val_indices, test_indices])):
    logging.info('\nChromosomes for {} ({}) - contain {} seqs:'.format(n, c, len(indices[i])))
    logging.info('{} - promoter active\n{} - nonpromoter active\n{} - promoter inactive\n{} - nonpromoter inactive'
          .format(data_labels[i][0], data_labels[i][1], data_labels[i][2], data_labels[i][3]))
    # Writing IDs for each split into file
    with open(os.path.join(output, '{}_{}.txt'.format(namespace, n)), 'w') as f:
        for j in ind:
            f.write(dataset.IDs[j] + '\n')

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

logging.info('\nTraining, validation and testing datasets built in {:.2f} s'.format(time() - t0))

num_batches = math.ceil(train_len / batch_size)

model = network(seq_len)
optimizer = optim_method(model.parameters(), lr=0.01, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()
best_acc = 0.0
logging.info('\n--- TRAINING ---')
t = time()
for epoch in range(num_epochs):
    t0 = time()
    model.train()
    confusion_matrix = np.zeros((num_classes, num_classes))
    train_loss = 0.0
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

        train_loss += loss.cpu().data
        _, indices = torch.max(outputs, axis=1)
        for ind, label in zip(indices, labels.cpu()):
            confusion_matrix[ind][label] += 1

        if i % 10 == 0:
            logging.info('Epoch {}, batch {}/{}'.format(epoch+1, i, num_batches))

    # Call the learning rate adjustment function
    if not args.no_adjust_lr:
        adjust_learning_rate(epoch, optimizer)

    train_sens, train_spec = [], []
    for cl in range(num_classes):
        tp = confusion_matrix[cl][cl]
        fn = sum([confusion_matrix[row][cl] for row in range(num_classes) if row != cl])
        tn = sum([confusion_matrix[row][col] for row, col in product(range(num_classes), range(num_classes))
                  if row != cl and col != cl])
        fp = sum([confusion_matrix[cl][col] for col in range(num_classes) if col != cl])
        train_sens += [float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0]
        train_spec += [float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0]
    train_loss = train_loss / num_batches

    with torch.set_grad_enabled(False):
        model.eval()
        confusion_matrix = np.zeros((num_classes, num_classes))
        for i, (seqs, labels) in enumerate(val_loader):

            if use_cuda:
                seqs = seqs.cuda()
                labels = labels.cuda()
            seqs = seqs.float()
            labels = labels.long()

            outputs = model(seqs)

            _, indices = torch.max(outputs, axis=1)
            for ind, label in zip(indices, labels.cpu()):
                confusion_matrix[ind][label] += 1

        val_sens, val_spec = [], []
        for cl in range(num_classes):
            tp = confusion_matrix[cl][cl]
            fn = sum([confusion_matrix[row][cl] for row in range(num_classes) if row != cl])
            tn = sum([confusion_matrix[row][col] for row, col in product(range(num_classes), range(num_classes))
                      if row != cl and col != cl])
            fp = sum([confusion_matrix[cl][col] for col in range(num_classes) if col != cl])
            val_sens += [float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0]
            val_spec += [float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0]

    # Save the model if the test acc is greater than our current best
    if mean(val_sens) > best_acc:
        torch.save(model.state_dict(), os.path.join(output, "{}_{}.model".format(namespace, epoch+1)))
        best_acc = mean(val_sens)

    # Print the metrics
    logging.info("Epoch {} finished in {:.2f} min\nTrain loss: {:1.3f}\n{:>35s}{:.5s}, {:.5s}"
                 .format(epoch+1, (time() - t0)/60, train_loss, '', 'SENSITIVITY', 'SPECIFICITY'))
    logging.info("--{:>18s} :{:>5} seqs{:>15}".format('TRAINING', train_len, "--"))
    for cl, seqs, sens, spec in zip(dataset.classes,data_labels[0], train_sens, train_spec):
        logging.info('{:>20} :{:>5} seqs - {:1.3f}, {:1.3f}'.format(cl, seqs, sens, spec))
    logging.info("--{:>18s} :{:>5} seqs{:>15}".format('VALIDATION', val_len, "--"))
    for cl, seqs, sens, spec in zip(dataset.classes, data_labels[1], val_sens, val_spec):
        logging.info('{:>20} :{:>5} seqs - {:1.3f}, {:1.3f}'.format(cl, seqs, sens, spec))
    logging.info(
        "--{:>18s} : {:1.3f}, {:1.3f}{:>12}".format('TRAINING MEANS', *list(map(mean, [train_sens, train_spec])), "--"))
    logging.info(
        "--{:>18s} : {:1.3f}, {:1.3f}{:>12}\n\n".format('VALIDATION MEANS', *list(map(mean, [val_sens, val_spec])), "--"))

    if mean(val_sens) >= acc_threshold:
        logging.info('Validation accuracy threshold reached!')
        break

logging.info('Training for {} finished in {:.2f} min'.format(namespace, (time() - t)/60))
