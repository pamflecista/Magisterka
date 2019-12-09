from statistics import mean
from itertools import product
import os


def make_chrstr(chrlist):

    cl = chrlist.copy()
    cl.append(0)
    chrstr = ''
    first = cl[0]
    och = first
    for ch in cl:
        if ch == och:
            och += 1
        elif first != och-1:
            if len(chrstr) != 0:
                chrstr += ', '
            chrstr += '%d-%d' % (first, och-1)
            first = ch
            och = ch+1
        else:
            if len(chrstr) != 0:
                chrstr += ', '
            chrstr += '%d' % first
            first = ch
            och = ch+1

    return chrstr


def read_chrstr(chrstr):

    chrstr = chrstr.strip('[]')
    c = chrstr.split(',')
    chrlist = []
    for el in c:
        el = el.split('-')
        if len(el) == 1:
            chrlist.append(int(el[0]))
        else:
            chrlist += [i for i in range(int(el[0]), int(el[1])+1)]
    chrlist.sort()

    return chrlist


def calculate_metrics(confusion_matrix, losses):
    num_classes = confusion_matrix.shape[0]
    sens, spec = [], []
    for cl in range(num_classes):
        tp = confusion_matrix[cl][cl]
        fn = sum([confusion_matrix[row][cl] for row in range(num_classes) if row != cl])
        tn = sum([confusion_matrix[row][col] for row, col in product(range(num_classes), repeat=2)
                  if row != cl and col != cl])
        fp = sum([confusion_matrix[cl][col] for col in range(num_classes) if col != cl])
        sens += [float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0]
        spec += [float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0]
    loss = [mean(el) if el else None for el in losses]
    return loss, sens, spec


def write_params(params, glob, file):
    with open(file, 'w') as f:
        for name, value in params.items():
            v = glob[value]
            if isinstance(v, list):
                if 'chr' in value:
                    f.write('{}: {}\n'.format(name, make_chrstr(v)))
                else:
                    f.write('{}: {}\n'.format(name, '; '.join(list(map(str, v)))))
            else:
                f.write('{}: {}\n'.format(name, v))


def read_classes(file):
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('Possible classes'):
                neurons = line.split(':')[1].strip().split('; ')
                break
    return neurons


def basic_params(parser, plotting=False):
    parser.add_argument('-p', '--path', action='store', metavar='DIR', type=str, default=None,
                        help='Working directory.')
    parser.add_argument('--namespace', action='store', metavar='NAME', type=str, default=None,
                        help='Namespace of the analysis, default: established based on input file')
    parser.add_argument('-o', '--output', action='store', metavar='DIR', type=str, default=None,
                        help='Output directory, default: [PATH]/results/[NAMESPACE]')
    parser.add_argument('--seed', action='store', metavar='NUMBER', type=int, default='0',
                        help='Set random seed, default: 0')
    if plotting:
        parser.add_argument('--param', action='store', metavar='NAME', type=str, default=None,
                            help='File with parameters of the network, from which results should be plotted, ' +
                                 'if PATH is given, file is supposed to be in PATH directory: [PATH]/[NAME], ' +
                                 'default: [PATH]/[NAMESPACE]_params.txt')
    return parser


def parse_arguments(args, file, namesp=None):
    if args.path is not None:
        path = args.path
    elif isinstance(file, list):
        path = '/' + '/'.join(file[0].strip('/').split('/')[:-1])
        for f in file[1:]:
            p = '/' + '/'.join(f.strip('/').split('/')[:-1])
            path = ''.join([el for el, le in zip(p, path) if el == le])
    else:
        path = '/' + '/'.join(file.strip('/').split('/')[:-1])
    if path.endswith('data'):
        path = path[:-4]
    if args.namespace is not None:
        namespace = args.namespace
    elif namesp is not None:
        namespace = namesp
    elif file is not None:
        namespace = file[0].strip('/').split('/')[-1].split('_')[0]
    else:
        namespace = path.strip('/').split('/')[-1]
    if args.output is not None:
        output = args.output
    else:
        if 'results' in path:
            output = path
        else:
            output = os.path.join(path, 'results', namespace)
    return path, output, namespace, args.seed


def write_results(logger, columns, variables, epoch):
    for stage in ['train', 'val']:
        result_string = '{}\t{}'.format(epoch, stage)
        for col, formatting in columns:
            if formatting == 'float-list':
                result_string += '\t' + ', '.join(['{:.2f}'.format(el) for el in variables['{}_{}'.format(stage, col)]])
    logger.info(result_string)


def print_results(logger, columns, variables, epoch):
    logger.info("Epoch {} finished in {:.2f} min\nTrain loss: {:1.3f}\n{:>35s}{:.5s}, {:.5s}"
                .format(epoch + 1, (time() - t0) / 60, train_loss_reduced, '', 'SENSITIVITY', 'SPECIFICITY'))
    logger.info("--{:>18s} :{:>5} seqs{:>15}".format('TRAINING', train_len, "--"))
    for cl, seqs, sens, spec in zip(dataset.classes, data_labels[0], train_sens, train_spec):
        logger.info('{:>20} :{:>5} seqs - {:1.3f}, {:1.3f}'.format(cl, seqs, sens, spec))
    logger.info("--{:>18s} :{:>5} seqs{:>15}".format('VALIDATION', val_len, "--"))
    for cl, seqs, sens, spec in zip(dataset.classes, data_labels[1], val_sens, val_spec):
        logger.info('{:>20} :{:>5} seqs - {:1.3f}, {:1.3f}'.format(cl, seqs, sens, spec))
    logger.info(
        "--{:>18s} : {:1.3f}, {:1.3f}{:>12}".format('TRAINING MEANS', *list(map(mean, [train_sens, train_spec])), "--"))
    logger.info(
        "--{:>18s} : {:1.3f}, {:1.3f}{:>12}\n\n".format('VALIDATION MEANS', *list(map(mean, [val_sens, val_spec])),
                                                        "--"))
