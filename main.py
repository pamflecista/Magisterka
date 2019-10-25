from sklearn.preprocessing import OneHotEncoder as Encoder
import numpy as np


def one_hot_encode(seqs):
    n = np.array(['A', 'C', 'T', 'G'])
    encoder = Encoder(sparse=False, categories=[n])
    encoder.fit(n.reshape(-1, 1))
    batch = None
    for seq in seqs:
        s = np.array([el for el in seq]).reshape(-1, 1)
        ohe = encoder.transform(s)
        if batch is None:
            batch = torch.Tensor([ohe])
        else:
            batch = torch.concatenate(batch, ohe)
