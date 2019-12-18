import argparse
from bin.funcs import *
from bin.datasets import *
import torch
from train import NET_TYPES
from torch.utils.data.sampler import SubsetRandomSampler

parser = argparse.ArgumentParser(description='Test given network based on given data')
parser.add_argument('--model', action='store', metavar='NAME', type=str, default=None,
                    help='File with the model to test')
parser = basic_params(parser)
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, args.model)

if args.model is None:
    modelfile = os.path.join(path, '{}_last.model'.format(namespace))
elif not args.model.startswith(path):
    modelfile = args.model
else:
    modelfile = os.path.join(path, args.model)

with open(os.path.join(path, '{}_params.txt'.format(namespace)), 'r') as f:
    for line in f:
        if line.startswith('Network type'):
            network = NET_TYPES[line.split(':')[-1].strip().lower()]
        elif line.startswith('Data directory'):
            data_dir = line.split(':')[-1].strip()
        elif line.startswith('Batch size'):
            batch_size = int(line.split(':')[-1].strip())

seq_len = 2000
model = network(seq_len)
model.load_state_dict(torch.load(modelfile))
model.eval()

dataset = SeqsDataset(data_dir, seq_len=seq_len)
num_classes = len(dataset.classes)
indices = list(map(int, open(os.path.join(path, '{}_train.txt'.format(namespace)), 'r').read().strip().split('\n')))
sampler = SubsetRandomSampler(indices)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

val_loss_neurons = [[] for _ in range(num_classes)]
true, scores = [], []
confusion_matrix = np.zeros((num_classes, num_classes))
output_values = [[[] for _ in range(num_classes)] for _ in range(num_classes)]
for i, (seqs, labels) in enumerate(loader):
    if use_cuda:
        seqs = seqs.cuda()
        labels = labels.cuda()
    seqs = seqs.float()
    labels = labels.long()

    outputs = model(seqs)

    for o, l in zip(outputs, labels):
        val_loss_neurons[l].append(-math.log((math.exp(o[l])) / (sum([math.exp(el) for el in o]))))

    _, indices = torch.max(outputs, axis=1)
    for ind, label, outp in zip(indices, labels.cpu(), outputs):
        confusion_matrix[ind][label] += 1
        output_values[label] = [el + [outp[j].cpu()] for j, el in enumerate(output_values[label])]

    true += labels.tolist()
    scores += outputs.tolist()

# Calculate metrics
val_losses, val_sens, val_spec = calculate_metrics(confusion_matrix, val_loss_neurons)
val_auc = calculate_auc(true, scores, num_classes)
