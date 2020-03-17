import argparse
from bin.common import *
from bin.datasets import SeqsDataset

parser = argparse.ArgumentParser(description='Get given number of the best and the worst sequences from each class '
                                             'for given model')
parser.add_argument('--num_seq', action='store', metavar='NUMBER', type=int, default=1,
                    help='Number of the best and the worst sequences to get, default value is 1')
group1 = parser.add_mutually_exclusive_group(required=False)
group1.add_argument('--test', action='store_true',
                    help='Get extreme sequences from the test data of the given model, by default training sequences are used')
group1.add_argument('--valid', action='store_true',
                    help='Get extreme sequences from the valid data of the given model, by default training sequences are used')
parser = basic_params(parser, param=True)
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, None)

if args.param is None:
    param_file = os.path.join(path, 'results', namespace, '{}_params.txt'.format(namespace))
else:
    param_file = args.param

network, data_dir, seq_len, ch, classes, name = params_from_file(param_file)
dataset = SeqsDataset(data_dir, seq_len=seq_len)

if args.test:
    seq_order = open(os.path.join(path, 'results', namespace, '{}_test.txt'.format(namespace)), 'r').read().strip().split('\n')
    outputs = np.load(os.path.join(path, 'results', namespace, '{}_test_outputs.npy'.format(namespace)), allow_pickle=True)
elif args.valid:
    seq_order = open(os.path.join(path, 'results', namespace, '{}_valid.txt'.format(namespace)), 'r').read().strip().split('\n')
    outputs = np.load(os.path.join(path, 'results', namespace, '{}_valid_outputs.npy'.format(namespace)), allow_pickle=True)
else:
    seq_order = open(os.path.join(path, 'results', namespace, '{}_train.txt'.format(namespace)), 'r').read().strip().split('\n')
    outputs = np.load(os.path.join(path, 'results', namespace, '{}_train_outputs.npy'.format(namespace)), allow_pickle=True)

labels = [dataset.__getitem__(el, info=True)[3] for el in seq_order]

subset = 'test' if args.test else 'valid' if args.valid else 'train'
result_file = os.path.join(output, 'extreme_{}_{}_{}.fasta'.format(namespace, subset, args.num_seq))
result_fasta = open(result_file, 'w')
for i, cl in enumerate(classes):
    print('class {}'.format(cl))
    r = [[j, el] for j, el in enumerate(outputs[i][i])]  # output values from sequences from the given class
    r.sort(key=lambda x: x[1])  # sorting - checking which seqs get the best/worst value
    order = [el[0] for el in r]  # order of the seqs from the cl class by the output value from the network
    class_index = [j for j, el in enumerate(labels) if el == i]  # which seqs in the list are from the cl class
    class_seq = np.array(seq_order)[class_index]  # get names of the seqs only from cl class in the correct order
    best = class_seq[order[-args.num_seq:]]  # get given number of the best seqs from cl class
    worst = class_seq[order[:args.num_seq]]  # get given number of the worst seqs from cl class
    for j, s in enumerate(best):
        ch, midpoint, strand, label, seq, _ = dataset.__getitem__(s, info=True)
        assert label == i
        result_fasta.write('> chr{} {} {} {} best{}\n{}'.format(ch, midpoint, strand, cl, j+1, seq))
    for j, s in enumerate(worst):
        ch, midpoint, strand, label, seq, _ = dataset.__getitem__(s, info=True)
        assert label == i
        result_fasta.write('> {} {} {} {} worst{}\n{}\n'.format(ch, midpoint, strand, cl, j+1, seq))
print('{} the best and {} the worst sequence(s) were written into {}'.format(args.num_seq, args.num_seq, result_file))

