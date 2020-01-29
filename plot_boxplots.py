import argparse
from bin.common import *

parser = argparse.ArgumentParser(description='Plot results based on given table')
parser.add_argument('-t', '--table', action='store', metavar='NAME', type=str, default=None,
                    help='Results table with data to plot, if PATH is given, file is supposed to be '
                         'in PATH directory: [PATH]/[NAME], default: [PATH]/[NAMESPACE]_results.tsv')
parser = basic_params(parser, plotting=True)
parser.add_argument('-c', '--column', action='store', metavar='COL', nargs='+', type=str, default=['loss'],
                    help='Number or name of column(s) to plot, default: loss')
group1 = parser.add_mutually_exclusive_group(required=False)
group1.add_argument('--train', action='store_true',
                    help='Use values from training, default values from validation are used')
group1.add_argument('--test', action='store_true',
                    help='Use testing results.')
parser.add_argument('--not_valid', action='store_true',
                    help='Do not print values from validation')
parser.add_argument('--print_mean', action='store_true',
                    help='Print also mean of the given data')
args = parser.parse_args()

