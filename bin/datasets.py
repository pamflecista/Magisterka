from .common import OHEncoder
from torch.utils.data import Dataset
import torch
import os
from .exceptions import *
from warnings import warn
from rewrite_fasta import rewrite_fasta
import math


class SeqsDataset(Dataset):

    def __init__(self, data, subset=(), filetype='fasta', seq_len=2000):

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
        locs = {}  # seq-name : index of element in dirs from which it was obtained
        for dd in data:
            if os.path.isfile(dd) and dd.endswith(filetype):
                name, _ = os.path.splitext(dd)
                if not subset or name in subset:
                    ids.append(name)
                    d = '/'.join(dd.split('/')[:-1])
                    dirs.append(d)
                    if name not in locs:
                        locs[name] = dirs.index(d)
                    else:
                        RepeatedFileError(name, dirs[locs[name]], d)
            for r, _, f in os.walk(dd, followlinks=True):
                fs = [el for el in f if el.endswith(filetype)]
                if len(fs) > 0:
                    if r not in dirs:
                        dirs.append(r)
                for file in fs:
                    name, _ = os.path.splitext(file)
                    if subset and name not in subset:
                        continue
                    ids.append(name)
                    if name not in locs:
                        locs[name] = dirs.index(r)
                    else:
                        raise RepeatedFileError(name, dirs[locs[name]], r)
        if len(ids) == 0:
            warn('No files of {} type was found in the given data'.format(filetype))
        self.IDs = ids
        self.locs = locs
        self.dirs = dirs
        self.filetype = filetype
        self.classes = ['promoter active', 'nonpromoter active', 'promoter inactive', 'nonpromoter inactive']
        self.num_classes = len(self.classes)
        self.num_seqs = len(self.IDs)
        self.seq_len = seq_len
        self.encoder = OHEncoder()

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index, info=False):
        try:
            ID = self.IDs[int(index)]
        except ValueError:
            ID = index
        filename = os.path.join(self.dirs[self.locs[ID]], '{}.{}'.format(ID, self.filetype))
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('>'):
                    ch, midpoint, strand, t1, t2 = line.strip('\n> ').split(' ')
                    label = self.classes.index('{} {}'.format(t1, t2))
                elif line:
                    seq = line.strip().upper()
                    if len(seq) > self.seq_len:
                        seq = seq[len(seq) // 2 - math.ceil(self.seq_len / 2): len(seq) // 2 + math.floor(self.seq_len / 2)]
                    else:
                        assert len(seq) == self.seq_len, 'Sequence {}: length {}'.format(self.IDs[index], len(seq))
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
        for i in range(self.__len__()):
            c, _, _, label, seq = self.__getitem__(i, info=True)
            ch = int(c.strip('chr').replace('X', '23').replace('Y', '23'))
            for j, chr_list in enumerate(chr_lists):
                if ch in chr_list:
                    indices[j].append(i)
        return indices

    def get_classes(self, indices=None):
        if indices is None:
            indices = [i for i in range(self.__len__())]
        result = {el: [] for el in self.classes}
        for i in indices:
            _, y = self.__getitem__(i)
            result[self.classes[y]].append(i)
        return result
