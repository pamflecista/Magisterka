import numpy as np
import torch
import random


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
            base = torch.from_numpy(baseline[i])
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
