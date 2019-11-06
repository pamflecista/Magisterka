from bin.datasets import SeqsDataset
import torch
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Train network based on given data')
parser.add_argument('data', action='store', metavar='DIR', type=str,
                    help='Directory of data for training, validation and testing.')
parser.add_argument('-train', action='store', metavar='NUMBER', type=str, default=-11,
                    help='Chromosome(s) for training, if negative it means the number of chromosomes '
                         'which should be randomly chosen. Default = -11')
parser.add_argument('-val', action='store', metavar='NUMBER', type=str, default=-9,
                    help='Chromosome(s) for validation, if negative it means the number of chromosomes '
                         'which should be randomly chosen. Default = -9')
parser.add_argument('-test', action='store', metavar='NUMBER', type=str, default=-3,
                    help='Chromosome(s) for testing, if negative it means the number of chromosomes '
                         'which should be randomly chosen. Default = -3')



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if use_cuda:
    print('--- CUDA available ---')
else:
    print('--- CUDA not available ---')

batch_size = 64
shuffle = True,
num_workers = 6
max_epochs = 100
validation_split = 0.1
dirs = './Data'
random_seed = 42

dataset = SeqsDataset(dirs)

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in train_loader:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_loader:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)



