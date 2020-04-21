import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from bin.common import basic_params, parse_arguments

COLORS = ['C{}'.format(i) for i in range(10)]

parser = argparse.ArgumentParser(description='Plot results based on given table')
parser.add_argument('-f', '--file', action='store', metavar='NAME', type=str, default=None, nargs='+',
                    help='Files with the outputs to plot, if PATH is given, file is supposed to be '
                         'in PATH directory: [PATH]/[NAME], default: [PATH]/[NAMESPACE]_outputs.npy')
parser = basic_params(parser, param=True)
args = parser.parse_args()

