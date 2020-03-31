import numpy as np
import torch
import random
import os


def integrated_gradients(model, inputs, labels, baseline=None, num_trials=10, steps=50, use_cuda=False):
    all_intgrads = []
    for i in range(num_trials):
        print('Trial {}'.format(i))
        if baseline is None:
            from .common import OHEncoder
            encoder = OHEncoder()
            base = torch.zeros(inputs.shape)
            for j, el in enumerate(inputs):
                seq = random.choices(encoder.dictionary, k=el.shape[-1])
                base[j] = torch.tensor(encoder(seq))
        elif type(baseline) is np.ndarray:
            base = torch.from_numpy(baseline[i]).reshape(inputs.shape)
        scaled_inputs = [base + (float(i) / steps) * (inputs - base) for i in range(0, steps + 1)]
        grads = calculate_gradients(model, scaled_inputs, labels, use_cuda=use_cuda)
        avg_grads = np.average(grads[:-1], axis=0)
        integrated_grad = (inputs - base) * torch.tensor(avg_grads)
        all_intgrads.append(integrated_grad)
    avg_intgrads = np.average(np.stack(all_intgrads), axis=0)
    return avg_intgrads


def calculate_gradients(model, inputs, labels, use_cuda=False):
    torch_device = [torch.device('cuda:0') if use_cuda else torch.device('cpu')][0]
    gradients = []
    for inp, label in zip(inputs, labels):
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
    from bin.common import OHEncoder
    from itertools import product
    from math import ceil
    print('Establishing random baseline named {}, num_seq = {}, n = {}'.format(name, num_seq, n))
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
    baseline_file = os.path.join(outdir, '{}_balanced_baseline_{}-{}.npy'.format(name, num_seq, n))
    np.save(baseline_file, base)
    print('Balanced baseline size {} was written into {}'.format(base.shape, baseline_file))
    return baseline_file
