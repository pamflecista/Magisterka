from bin.integrated_gradients import *
from bin.datasets import *
import pytest
from bin.networks import TestNetwork

@pytest.mark.parametrize("input_dir", ['/test/data/test.fasta'])
def test_integrated_gradients(input_dir):
    dataset = SeqsDataset(os.path.join('/home/marni/PycharmProjects/magisterka', input_dir), seq_len=10)
    X, y = dataset.__getitem__(0)
    labels = [y]
    X = [X]
    for i in range(1, len(dataset)):
        xx, yy = dataset.__getitem__(i)
        X.append(xx)
        labels.append(yy)
    X = torch.stack(X, dim=0)
    model = TestNetwork(10)
    results = integrated_gradients(model, X, 0)
    #results = calculate_gradients(X, model, labels)
    print(results)


@pytest.mark.parametrize("num_seq, n", [(16, 2)])
def test_produce_balanced_baseline(num_seq, n):
    outdir = '/home/marni/Dokumenty/magisterka/'
    f = produce_balanced_baseline(outdir, 'test', num_seq, n=n)
    array = np.load(f)
    print(array.shape)
    encoder = OHEncoder()
    dif = [set([]) for _ in range(array.shape[1])]
    for base in array:
        dif = [dif[i] | {encoder.decode(el[:, :3])} for i, el in enumerate(base)]
    for el in dif:
        assert len(el) == 4**n


@pytest.mark.parametrize("num_seq, n", [(20, 3)])
def test_produce_morebalanced_baseline(num_seq, n):
    outdir = '/home/marni/Dokumenty/magisterka/'
    f = produce_morebalanced_baseline(outdir, 'test', num_seq, n=n)
    array = np.load(f)
    print(array.shape)
