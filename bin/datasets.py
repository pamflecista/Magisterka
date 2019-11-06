from sklearn.preprocessing import OneHotEncoder as Encoder
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from Bio.Seq import Seq
from Bio import SeqIO
import sys
import torch
from os import listdir
from os import path
from .exceptions import *


class SeqsDataset(Dataset):

    def __init__(self, dirs, IDs=None, filetype='fasta'):
        if isinstance(dirs, str):
            dirs = [dirs]
        if len(dirs) == 1:
            dirs = [path.join(dirs[0], o) for o in listdir(dirs[0])
                    if path.isdir(path.join(dirs[0], o))]
        self.dirs = dirs

        # Establishing files' IDs and their directories
        ids, locs = [], {}
        if IDs is None:
            ids = []
            for i, directory in enumerate(dirs):
                for f in listdir(directory):
                    if path.isfile(path.join(directory, f)) and f.endswith(filetype):
                        name = '{}/{}'.format(directory.split('/')[-1], f.strip('.{}'.format(filetype)))
                        if name not in locs:
                            locs[name] = i
                        else:
                            raise RepeatedFileError(name, dirs[locs[name]], directory)
                        ids.append(name)
        else:
            for f, d in IDs.items():
                if f.endswith(filetype):
                    name = '{}/{}'.format(dirs[d].split('/')[-1], f.strip('.{}'.format(filetype)))
                    if name not in locs:
                        locs[name] = d
                    else:
                        raise RepeatedFileError(name, dirs[locs[name]], dirs[d])
                    ids.append(name)
                else:
                    raise TypeError

        self.IDs = ids
        self.locs = locs
        self.filetype = filetype
        self.label_coding = {'promoter': 1, 'nonpromoter': 0, 'active': 1, 'inactive': 0}
        self.encoder = OHEncoder()

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index, info=False):
        ID = self.IDs[index]
        with open(path.join(self.dirs[self.locs[ID]], ID, self.filetype), 'r') as file:
            for line in file:
                if line.startswith('>'):
                    ch, strand, t1, t2 = line.strip('\n> ').split(' ')
                    label = [self.label_coding[t1], self.label_coding[t2]]
                elif line:
                    seq = line.strip().upper()
                    break
        if info:
            return ch, strand, label, seq
        X = torch.Tensor(self.encoder(seq))
        y = torch.Tensor(label)
        return X, y

    def get_chr(self, ch):
        indeces = []
        labels = {}
        for i in range(self.__len__()):
            c, _, label, _ = self.__getitem__(i, info=True)
            if int(c.strip('ch')) == ch:
                indeces.append(i)
                l = '{}{}'.format(*label)
                labels[l] = labels.setdefault(l, 0) + 1
        return indeces, labels


class OHEncoder:

    def __init__(self, categories=np.array(['A', 'C', 'G', 'T'])):
        self.encoder = Encoder(sparse=False, categories=[categories])
        self.encoder.fit(categories.reshape(-1, 1))

    def __call__(self, seq):
        s = np.array([el for el in seq]).reshape(-1, 1)
        return self.encoder.transform(s)
