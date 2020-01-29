from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import argparse
from bin.common import *
import shutil
from warnings import warn
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from statistics import mean
import os


CLASSES = [
    'promoter active',
    'nonpromoter active',
    'promoter inactive',
    'nonpromoter inactive'
]

ENCODER = OHEncoder()


def auc_first(model, X, y):
    logger.info('Calculating AUC for first class')
    y_true = np.array([1 if el == 0 else 0 for el in y])
    return roc_auc_score(y_true, model.predict_proba(X)[:, 0])


def auc_second(model, X, y):
    logger.info('Calculating AUC for second class')
    y_true = np.array([1 if el == 1 else 0 for el in y])
    return roc_auc_score(y_true, model.predict_proba(X)[:, 1])


def auc_third(model, X, y):
    logger.info('Calculating AUC for third class')
    y_true = np.array([1 if el == 2 else 0 for el in y])
    return roc_auc_score(y_true, model.predict_proba(X)[:, 2])


def auc_fourth(model, X, y):
    logger.info('Calculating AUC for fourth class\n')
    y_true = np.array([1 if el == 3 else 0 for el in y])
    return roc_auc_score(y_true, model.predict_proba(X)[:, 3])


SCORERS = {
    'AUC1': auc_first,
    'AUC2': auc_second,
    'AUC3': auc_third,
    'AUC4': auc_fourth
}

parser = argparse.ArgumentParser(description='Train and test Random Forest classifier based on the given data')
parser.add_argument('data', action='store', metavar='DATASET', type=str, nargs='+',
                    help='Folder with the data for training and validation, if PATH is given, data is supposed to be ' +
                         'in PATH directory: [PATH]/data/[DATA]')
parser = basic_params(parser)
parser.add_argument('--run', action='store', metavar='NUMBER', type=str, default='0',
                    help='number of the analysis, by default NAMESPACE is set to [NETWORK][RUN]')
parser.add_argument('--cv', action='store', metavar='NUMBER', type=int, default=10,
                    help='Number of folds for cross-validation')
args = parser.parse_args()

path, output, namespace, seed = parse_arguments(args, args.data, namesp='forest{}'.format(args.run))
cv = args.cv
# create folder for the output files
if os.path.isdir(output):
    shutil.rmtree(output)
try:
    os.mkdir(output)
except FileNotFoundError:
    os.mkdir(os.path.join(path, 'results'))
    os.mkdir(output)
# establish data directories
if args.path is not None:
    data_dir = [os.path.join(path, 'data', d) for d in args.data]
else:
    data_dir = args.data
    if os.path.isdir(data_dir[0]):
        path = data_dir[0]

# Define files for logs and for results
(logger, results_table), old_results = build_loggers('cv', output=output, namespace=namespace)
logger.info('\nAnalysis {} begins {}\nInput data: {}\nOutput directory: {}\n'.format(
    namespace, datetime.now().strftime("%d/%m/%Y %H:%M:%S"), '; '.join(data_dir), output))

if not old_results:
    auc_header = '\t'.join(['AUC - {}'.format(c) for c in CLASSES])
    results_table.info('Dataset\tCV-fold\t{}'.format(auc_header))


def read_seq(file, X, y):
    f = open(file, 'r')
    for line in f:
        if line.startswith('>'):
            ch, midpoint, strand, t1, t2 = line.strip('\n> ').split(' ')
            label = CLASSES.index('{} {}'.format(t1, t2))
        elif line:
            seq = line.strip().upper()
    if f.readline().strip():
        warn('In file {} is more than one sequence!'.format(file))
    x = ENCODER(seq).flatten()
    if X.shape[0] == 0:
        X = np.array([x])
    else:
        X = np.vstack((X, x))
    y = np.append(y, label)
    return X, y


logger.info('Reading sequences from the files')
filetype = 'fasta'
X = np.array([])
y = np.array([])
for dd in data_dir:
    if os.path.isfile(dd) and dd.endswith(filetype):
        X, y = read_seq(dd, X, y)
    for r, _, f in os.walk(dd, followlinks=True):
        fs = [el for el in f if el.endswith(filetype)]
        for file in fs:
            X, y = read_seq(os.path.join(r, file), X, y)

logger.info('Creating random forest')
clf = RandomForestClassifier(max_depth=2, random_state=seed)

logger.info('{}-fold CV of the random forest based on {} seqs'.format(cv, X.shape[0]))
scores = cross_validate(clf, X, y, scoring=SCORERS,
                        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed))

auc_result = []
logger.info('\nMean AUC based on {}-fold CV:'.format(cv))
for i, c in enumerate(CLASSES):
    score = scores['test_AUC{}'.format(i+1)]
    logger.info('\t{}: {:.3f}'.format(c, mean(score)))
    auc_result.append([str(round(el, 4)) for el in score])

results_table.info('{}\t{}\t{}'.format('; '.join(data_dir), cv, '\t'.join([', '.join(el) for el in auc_result])))
