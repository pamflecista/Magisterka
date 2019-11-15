from bin.datasets import SeqsDataset
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim import Adam
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
from bin.funcs import *
from bin.networks import BassetNetwork
import math
import os
from statistics import mean
import logging
from time import time

NET_TYPES = {
    'Basset': BassetNetwork
}


def adjust_learning_rate(epoch, optimizer):
    lr = 0.001

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
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
                    help='Output directory, default: [PATH]/results/')
parser.add_argument('-p', '--path', action='store', metavar='DIR', type=str, default=None,
                    help='Working directory, default: ./')
parser.add_argument('--namespace', action='store', metavar='NAME', type=str, default=None,
                    help='Namespace of the analysis, default: [NETWORK]')
parser.add_argument('--run', action='store', metavar='NUMBER', type=str, default='0',
                    help='Number of the analysis, by default NAMESPACE is set to [NETWORK][RUN]')
parser.add_argument('-b', '--batch_size', action='store', metavar='INT', type=int, default=64,
                    help='Size of the batch, default: 64')
parser.add_argument('--num_workers', action='store', metavar='INT', type=int, default=4,
                    help='How many subprocesses to use for data loading, default: 4')
parser.add_argument('--num_epochs', action='store', metavar='INT', type=int, default=500,
                    help='Maximum number of epochs to run, default: 500')
parser.add_argument('--acc_threshold', action='store', metavar='FLOAT', type=float, default=0.9,
                    help='Threshold of the validation accuracy - if gained training process stops, default: 0.9')
args = parser.parse_args()

batch_size, num_workers, num_epochs, acc_threshold = args.batch_size, args.num_workers, args.num_epochs, \
                                                     args.acc_threshold

network = NET_TYPES[args.network]
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
    output = os.path.join(path, 'results')
    if not os.path.isdir(output):
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

# Creating data indices for training, validation and test splits:
indices, data_labels, seq_len = dataset.get_chrs([train_chr, val_chr, test_chr])
train_indices, val_indices, test_indices = indices
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

num_batches = math.ceil(len(train_indices) / batch_size)

model = network(seq_len)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()
best_acc = 0.0
logging.info('\n--- Training ---')
t = time()
for epoch in range(num_epochs):
    t0 = time()
    model.train()
    train_acc = [0.0] * dataset.num_classes
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
            if ind == label:
                train_acc[label] += 1

        if i % 10 == 0:
            logging.info('Epoch {}, batch {}/{}'.format(epoch+1, i, num_batches))

    # Call the learning rate adjustment function
    adjust_learning_rate(epoch, optimizer)

    train_acc = [float(acc) / data_labels[0][i] if data_labels[0][i] > 0 else 0.0 for i, acc in enumerate(train_acc)]
    train_loss = train_loss / num_batches

    with torch.set_grad_enabled(False):
        model.eval()
        val_acc = [0.0] * dataset.num_classes
        for i, (seqs, labels) in enumerate(val_loader):

            if use_cuda:
                seqs = seqs.cuda()
                labels = labels.cuda()
            seqs = seqs.float()
            labels = labels.long()

            outputs = model(seqs)

            _, indices = torch.max(outputs, axis=1)
            for ind, label in zip(indices, labels.cpu()):
                if ind == label:
                    val_acc[label] += 1

        val_acc = [float(acc) / data_labels[1][i] if data_labels[0][i] > 0 else 0.0 for i, acc in enumerate(val_acc)]

    # Save the model if the test acc is greater than our current best
    if mean(val_acc) > best_acc:
        torch.save(model.state_dict(), os.path.join(output, "{}_{}.model".format(namespace, epoch+1)))
        best_acc = mean(val_acc)

    # Print the metrics
    logging.info("Epoch {} finished in {:.2f} min\n-- Train accuracy --".format(epoch+1, (time() - t0)/60))
    for cl, acc in zip(dataset.classes, train_acc):
        logging.info('{} - {:.3}'.format(cl, acc))
    logging.info("Mean train accuracy - {:.3}\nTrain loss: {:.3}\n-- Validation Accuracy --".format(mean(train_acc), train_loss))
    for cl, acc in zip(dataset.classes, val_acc):
        logging.info('{} - {:.3}'.format(cl, acc))
    logging.info("Mean validation accuracy - {:.3}\n".format(mean(val_acc)))

    if mean(val_acc) >= acc_threshold:
        logging.info('Validation accuracy threshold reached!')
        break

logging.info('Training for {} finished in {:.2f} min'.format(namespace, (time() - t)/60))
