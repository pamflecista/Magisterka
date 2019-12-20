import os
from bin.networks import *
from collections import OrderedDict

NET_TYPES = {
    'basset': BassetNetwork,
    'custom': CustomNetwork
}

RESULTS_COLS = OrderedDict({
    'Loss': ['losses', 'float-list'],
    'Sensitivity': ['sens', 'float-list'],
    'Specificity': ['spec', 'float-list'],
    'AUC-neuron': ['aucINT', 'float-list']
})


def make_chrstr(chrlist):

    cl = chrlist.copy()
    cl.sort()
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
    from statistics import mean
    from itertools import product
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


def calculate_auc(true, scores, num_classes):
    from sklearn.metrics import roc_auc_score
    auc = [[] for _ in range(num_classes)]
    for neuron in range(num_classes):
        y_true = [1 if el == neuron else 0 for el in true]
        y_score = [el[neuron] for el in scores]
        auc[neuron].append(roc_auc_score(y_true, y_score))
        for neg in [i for i in range(num_classes) if i != neuron]:
            y_help = [1 if el == neuron else 0 if el == neg else -1 for el in true]
            y_score = [el[neuron] for use, el in zip(y_help, scores) if use != -1]
            y_true = [el for el in y_help if el != -1]
            auc[neuron].append(roc_auc_score(y_true, y_score))
    return auc


def write_params(params, glob, file):
    with open(file, 'w') as f:
        for name, value in params.items():
            v = glob[value]
            if isinstance(v, list):
                if 'chr' in value:
                    f.write('{}: {}\n'.format(name, make_chrstr(v)))
                else:
                    f.write('{}: {}\n'.format(name, ''.join(['\n\t{}'.format(el) for el in list(map(str, v))])))
            elif isinstance(v, dict):
                towrite = '{}:'.format(name)
                for key, val in v.items():
                    if isinstance(val, list):
                        val = ', '.join(map(str, val))
                    towrite += '\n\t{}: {}'.format(key, val)
                f.write('{}\n'.format(towrite))
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


def results_header(stage, logger, classes=[]):
    if 'AUC-neuron' in RESULTS_COLS.keys():
        name, formatting = RESULTS_COLS['AUC-neuron']
        del RESULTS_COLS['AUC-neuron']
        for i, classname in enumerate(classes):
            RESULTS_COLS['AUC - {}'.format(classname)] = [name.replace('INT', str(i)), formatting]
    towrite = '\t'.join(RESULTS_COLS.keys())
    if stage == 'train':
        logger.info('Epoch\tStage\t{}'.format(towrite))
    elif stage == 'test':
        logger.info('Dataset\tSubset\t{}'.format(towrite))
    return logger


def write_train_results(logger, variables, epoch):
    for stage in ['train', 'val']:
        result_string = '{}\t{}'.format(epoch+1, stage)
        for col, formatting in RESULTS_COLS.values():
            if col[-1].isdigit():
                variable = variables['{}_{}'.format(stage, col[:-1])][int(col[-1])]
            else:
                variable = variables['{}_{}'.format(stage, col)]
            if formatting == 'float-list':
                result_string += '\t' + ', '.join(['{:.2f}'.format(el) for el in variable])
            elif formatting == 'float':
                result_string += '\t{:.2f}'.format(variables['{}_{}'.format(stage, col)])
        logger.info(result_string)


def write_test_results(logger, variables):
    pass


def check_cuda(logger):
    import torch
    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        logger.info('--- CUDA available ---')
    else:
        logger.info('--- CUDA not available ---')
    return use_cuda


def build_loggers(stage, output='./', namespace='test', verbose_mode=True, logfile=True, resultfile=True):
    import logging
    formatter = logging.Formatter('%(message)s')
    loggers = []
    if logfile or verbose_mode:
        logger = logging.getLogger('verbose')
        logger.setLevel(logging.INFO)
        loggers.append(logger)
    if verbose_mode:
        cmd_handler = logging.StreamHandler()
        cmd_handler.setFormatter(formatter)
        logger.addHandler(cmd_handler)
    if logfile:
        log_handler = logging.FileHandler(os.path.join(output, '{}_{}.log'.format(namespace, stage)))
        log_handler.setFormatter(formatter)
        logger.addHandler(log_handler)
    if resultfile:
        results_table = logging.getLogger('results')
        results_handler = logging.FileHandler(os.path.join(output, '{}_{}_results.tsv'.format(namespace, stage)))
        results_handler.setFormatter(formatter)
        results_table.addHandler(results_handler)
        results_table.setLevel(logging.INFO)
        loggers.append(results_table)
    return loggers


'''def print_results(logger, columns, variables, epoch):
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
    formatter = logging.Formatter('%(message)s')
logger = logging.getLogger('verbose')
results_table = logging.getLogger('results')
cmd_handler = logging.StreamHandler()
log_handler = logging.FileHandler(os.path.join(output, '{}.log'.format(namespace)))
results_handler = logging.FileHandler(os.path.join(output, '{}_results.tsv'.format(namespace)))
for logg, handlers in zip([logger, results_table], [[cmd_handler, log_handler], [results_handler]]):
    for handler in handlers:
        handler.setFormatter(formatter)
        logg.addHandler(handler)
    logg.setLevel(logging.INFO)                                         
                                                        
                                                        
                                                        '''