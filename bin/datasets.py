from sklearn.preprocessing import OneHotEncoder as Encoder
import numpy as np
from torch.utils.data import Dataset
import torch
import os
from .exceptions import *
from warnings import warn
from rewrite_fasta import rewrite_fasta


class SeqsDataset(Dataset):

    def __init__(self, data, filetype='fasta'):

        # Establishing files' IDs and their directories
        if isinstance(data, str):
            if data.endswith(filetype):
                i, path = rewrite_fasta(data)
                if i == 1:
                    warn('Only one sequence found in the given data!')
                data = [path]
            elif os.path.isdir(data):
                data = [data]
            else:
                raise GivenDataError(data, filetype)
        ids = []
        dirs = []
        locs = {}
        for dd in data:
            if os.path.isfile(dd) and dd.endswith(filetype):
                name = dd.strip('.{}'.format(filetype))
                ids.append(name)
                d = '/'.join(dd.split('/')[:-1])
                if name not in locs:
                    locs[name] = d
                else:
                    RepeatedFileError(name, dirs[locs[name]], d)
            for r, _, f in os.walk(dd):
                fs = [el for el in f if el.endswith(filetype)]
                if len(fs) > 0:
                    if r not in dirs:
                        dirs.append(r)
                for file in fs:
                    name = file.strip('.{}'.format(filetype))
                    ids.append(name)
                    if name not in locs:
                        locs[name] = dirs.index(r)
                    else:
                        RepeatedFileError(name, dirs[locs[name]], r)
        if len(ids) == 0:
            warn('No files of the {} type was found in the given data'.format(filetype))
        self.IDs = ids
        self.locs = locs
        self.dirs = dirs
        self.filetype = filetype
        self.classes = ['promoter active', 'nonpromoter active', 'promoter inactive', 'nonpromoter inactive']
        self.num_classes = len(self.classes)
        self.encoder = OHEncoder()

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index, info=False):
        ID = self.IDs[index]
        filename = os.path.join(self.dirs[self.locs[ID]], '{}.{}'.format(ID, self.filetype))
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    ch, midpoint, strand, t1, t2 = line.strip('\n> ').split(' ')
                    label = self.classes.index('{} {}'.format(t1, t2))
                elif line:
                    seq = line.strip().upper()
                    break
            if file.readline().strip():
                warn('In file {} is more than one sequence!'.format(filename))
        if info:
            return ch, midpoint, strand, label, seq
        X = torch.tensor(self.encoder(seq))
        X = X.reshape(1, *X.size())
        y = torch.tensor(label)
        return X, y

    def get_chrs(self, chr_lists):
        indices = [[] for _ in range(len(chr_lists))]
        labels = [{el: 0 for el in range(self.num_classes)} for _ in range(len(chr_lists))]
        seq_len = None
        for i in range(self.__len__()):
            c, _, _, label, seq = self.__getitem__(i, info=True)
            if seq_len is None:
                seq_len = len(seq)
            else:
                assert len(seq) == seq_len
            for j, chr_list in enumerate(chr_lists):
                if int(c.strip('chr').replace('X', '23').replace('Y', '23')) in chr_list:
                    indices[j].append(i)
                    labels[j][label] += 1
        return indices, labels, seq_len


class OHEncoder:

    def __init__(self, categories=np.array(['A', 'C', 'G', 'T'])):
        self.encoder = Encoder(sparse=False, categories=[categories])
        self.encoder.fit(categories.reshape(-1, 1))

    def __call__(self, seq):
        s = np.array([el for el in seq]).reshape(-1, 1)
        return self.encoder.transform(s).T
