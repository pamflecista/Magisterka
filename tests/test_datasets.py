from ..bin.datasets import *

def test_read_seqs(file='../data/test.fasta'):
    pass

def test_SeqsDataset():
    dataset = SeqsDataset('/home/marni/magisterka/data/dataset5/pa_specific_reference_met.fa')
    train_chr = [1, 2]
    val_chr = [3, 4]
    test_chr = [5, 6]
    indices = dataset.get_chrs([train_chr, val_chr, test_chr])
    class_stage = [dataset.get_classes(i) for i in indices]
