from bin.datasets import SeqsDataset
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim import Adam
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
from bin.funcs import *
from warnings import warn
from bin.networks import BassetNetwork
import math
import os


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
                    help='Directory of data for training and validation.')
parser.add_argument('-train', action='store', metavar='CHR', type=str, default='1-13',
                    help='Chromosome(s) for training, if negative it means the number of chromosomes '
                         'which should be randomly chosen. Default = 1-13')
parser.add_argument('-val', action='store', metavar='CHR', type=str, default='14-18',
                    help='Chromosome(s) for validation, if negative it means the number of chromosomes '
                         'which should be randomly chosen. Default = 14-18')
parser.add_argument('-test', action='store', metavar='CHR', type=str, default='19-22',
                    help='Chromosome(s) for testing, if negative it means the number of chromosomes '
                         'which should be randomly chosen. Default = 19-22')
parser.add_argument('-o', '--output', action='store', metavar='DIR', type=str, default='./',
                    help='Output directory')
parser.add_argument('-b', '--batch_size', action='store', metavar='INT', type=int, default=30,
                    help='Size of the batch, default is 30')
parser.add_argument('--num_workers', action='store', metavar='INT', type=int, default=4,
                    help='How many subprocesses to use for data loading, default is 4')
parser.add_argument('--num_epochs', action='store', metavar='INT', type=int, default=100,
                    help='Number of epochs to run, default is 100')

args = parser.parse_args()

output, batch_size, num_workers, num_epochs = args.output, args.batch_size, args.num_workers, args.num_epochs

train_chr = read_chrstr(args.train)
val_chr = read_chrstr(args.val)
if set(train_chr) & set(val_chr):
    warn('Chromosomes for training and chromosomes for validation overlap!')
data_dir = args.data

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    print('--- CUDA available ---')
else:
    print('--- CUDA not available ---')

dataset = SeqsDataset(data_dir)

# Creating data indices for training and validation splits:
indices, labels, seq_len = dataset.get_chrs([train_chr, val_chr])
train_indices, val_indices = indices
train_len, val_len = len(train_indices), len(val_indices)
for i, (n, c) in enumerate(zip(['training', 'validation'], [args.train, args.val])):
    print('\nChromosomes for {} ({}) - contain {} seqs:'.format(n, c, len(indices[i])))
    print('{} - active promoters\n{} - inactive promoters\n{} - active nonpromoters\n{} - inactive nonpromoters'
          .format(labels[i]['11'], labels[i]['10'], labels[i]['01'], labels[i]['00']))

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

num_batches = math.ceil(len(train_indices) / batch_size)

model = BassetNetwork(seq_len)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.MSELoss()
best_acc = 0.0
print('\n--- Training ---')
for epoch in range(num_epochs):
    model.train()
    train_acc = 0.0
    train_loss = 0.0
    for i, (seqs, labels) in enumerate(train_loader):
        if use_cuda:
            seqs = seqs.cuda()
            labels = labels.cuda()
            model.cuda()
        seqs = seqs.float()
        labels = labels.float()

        optimizer.zero_grad()
        outputs = model(seqs)
        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.cpu().data * seqs.size(0)

        train_acc += torch.sum(torch.tensor(list(map(round, map(float, outputs.flatten())))).reshape(outputs.shape) ==
                               labels.cpu().long())

        if i % 10 == 0:
            print('Epoch {}, batch {}/{}'.format(epoch+1, i, num_batches))

    # Call the learning rate adjustment function
    adjust_learning_rate(epoch, optimizer)

    train_acc = float(train_acc) / (train_len*2)
    train_loss = float(train_loss) / (train_len*2)

    with torch.set_grad_enabled(False):
        model.eval()
        val_acc = 0.0
        for i, (seqs, labels) in enumerate(val_loader):

            if use_cuda:
                seqs = seqs.cuda()
                labels = labels.cuda()
            seqs = seqs.float()
            labels = labels.float()

            outputs = model(seqs)

            val_acc += torch.sum(torch.tensor(list(map(round, map(float, outputs.flatten())))).reshape(outputs.shape) ==
                                 labels.cpu().long())

        val_acc = float(val_acc) / (val_len*2)

    # Save the model if the test acc is greater than our current best
    if val_acc > best_acc:
        torch.save(model.state_dict(), os.path.join(output, "BassetNetwork_{}.model".format(epoch)))
        best_acc = val_acc

    # Print the metrics
    print("Epoch {}, Train Accuracy: {:.3} , Train Loss: {:.3} , Test Accuracy: {:.3}".format(epoch+1, train_acc,
                                                                                              train_loss, val_acc))
