from sklearn.preprocessing import OneHotEncoder as Encoder
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from Bio.Seq import Seq
from Bio import SeqIO
import sys
import torch


class SeqsDataset(Dataset):

    def __init__(self, data, labels, names):
        self.data = data
        self.labels = labels
        self.names = names

    def __name__(self, n):
        return self.names[n]

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]

        return X, y


def one_hot_encode(seqs):
    n = np.array(['A', 'C', 'G', 'T'])
    encoder = Encoder(sparse=False, categories=[n])
    encoder.fit(n.reshape(-1, 1))
    batch = None
    for seq in seqs:
        s = np.array([el for el in seq]).reshape(-1, 1)
        ohe = encoder.transform(s)
        if batch is None:
            batch = torch.Tensor([ohe])
        else:
            batch = torch.cat((batch, torch.Tensor([ohe])))

    return batch.reshape((batch.shape[0], 1, -1, 4))


def read_seqs(file):
    with open(file, 'r') as handler:
        records = []
        for record in SeqIO.parse(handler, "fasta"):
            records.append(record)
    labels = []
    seqs = []
    for record in records:
        d = record.description.split(' ')
        strand = d[1]
        if strand == '+':
            seqs.append(str(record.seq).upper())
        elif strand == '-':
            seqs.append(str(record.seq.revers_complement()).upper())
        labels.append(torch.Tensor(list(map(float, d[2:4]))))
    return seqs, labels


def loading_data(file, bs=100):
    seqs, labels = read_seqs(file)
    loader = DataLoader(Dataset(one_hot_encode(seqs), labels), batch_size=bs)

