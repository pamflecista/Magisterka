import numpy as np
import torch
import random
import os
from bin.common import OHEncoder


def integrated_gradients(model, inputs, labels, baseline=None, num_trials=10, steps=50, use_cuda=False):
    all_integrads = []
    for i in range(num_trials):
        print('Trial {}'.format(i))
        if baseline is None:
            encoder = OHEncoder()
            base = torch.zeros(inputs.shape)
            for j, el in enumerate(inputs):
                seq = random.choices(encoder.dictionary, k=el.shape[-1])
                base[j] = torch.tensor(encoder(seq))
        elif type(baseline) is np.ndarray:
            base = torch.from_numpy(baseline[i]).reshape(inputs.shape)
        scaled_inputs = [base + (float(i) / steps) * (inputs - base) for i in range(1, steps + 1)]
        grads = calculate_gradients(model, scaled_inputs, labels, use_cuda=use_cuda)
        avg_grads = np.average(grads[:-1], axis=0)
        integrated_grad = (inputs - base) * torch.tensor(avg_grads)
        all_integrads.append(integrated_grad)
    avg_integrads = np.average(np.stack(all_integrads), axis=0)
    return avg_integrads


def calculate_gradients(model, inputs, labels, use_cuda=False):
    torch_device = [torch.device('cuda:0') if use_cuda else torch.device('cpu')][0]
    gradients = []
    for inp, label in zip(inputs, labels):
        model.eval()
        inp = inp.float()
        inp.to(torch_device)
        inp.requires_grad = True
        output = model(inp)
        gradient = []
        for i in range(output.shape[0]):
            model.zero_grad()
            output[i][label].backward(retain_graph=True)
            gradient.append(inp.grad[i].detach().cpu().numpy())
        gradients.append(gradient)
    gradients = np.array(gradients)
    return gradients


def produce_balanced_baseline(outdir, name, num_seq, n=3):
    from itertools import product
    from math import ceil
    print('Establishing balanced baseline named {}, num_seq = {}, n = {}'.format(name, num_seq, n))
    trials = 4**n
    encoder = OHEncoder()
    d = encoder.dictionary
    ntuples = [''.join(el) for el in product(d, repeat=n)]
    presence = [[[] for _ in range(ceil(2000/n))] for _ in range(num_seq)]
    base = []
    for _ in range(trials):
        b = np.zeros((num_seq, 4, 2000))
        for j in range(num_seq):
            choice = [random.choice([el for el in range(len(ntuples)) if el not in la]) for la in presence[j]]
            presence[j] = [la + [el] for la, el in zip(presence[j], choice)]
            seq = ''.join([ntuples[i] for i in choice])[:2000]
            b[j] = encoder(seq)
        base.append(b)
    base = np.stack(base)
    baseline_file = os.path.join(outdir, '{}-balanced-{}-{}_baseline.npy'.format(name, num_seq, n))
    np.save(baseline_file, base)
    print('Balanced baseline size {} was written into {}'.format(base.shape, baseline_file))
    return baseline_file


def produce_morebalanced_baseline(outdir, name, num_seq, n=3, same_for_each_seq=True):
    def kmers(k, sigma="ACGT"):
        if k == 1:
            for s in sigma:
                yield [s]
        else:
            kms = kmers(k - 1, sigma)
            for k in kms:
                for l in sigma:
                    yield k + [l]

    def generate_seqs(l=1, k=1, sigma="ACGT"):
        result = list(kmers(k, sigma))
        sigma = list(sigma)
        for i in range(max(0, l - k)):
            # extend the lists by one
            random.shuffle(sigma)
            result.sort()
            for i, r in enumerate(result):
                r.insert(0, sigma[i % len(sigma)])
        return ["".join(r) for r in result]

    trials = 4 ** n
    encoder = OHEncoder()
    d = encoder.dictionary
    if same_for_each_seq:
        ss = generate_seqs(l=2000, k=n, sigma=d)
        ss.sort()
        b = np.stack([np.array(encoder(seq)) for seq in ss], axis=0)
        base = np.stack([b for _ in range(num_seq)], axis=1)
    else:
        base = np.zeros((trials, num_seq, 4, 2000))
        for num in range(num_seq):
            ss = generate_seqs(l=2000, k=n, sigma=d)
            ss.sort()
            b = np.stack([np.array(encoder(seq)) for seq in ss], axis=0)
            base[:, num, :, :] = b
    baseline_file = os.path.join(outdir, '{}-morebalanced-{}-{}_baseline.npy'.format(name, num_seq, n))
    np.save(baseline_file, base)
    print('Balanced baseline size {} was written into {}'.format(base.shape, baseline_file))
    return baseline_file
