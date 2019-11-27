import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot results based on given table')
parser.add_argument('-f', '--file', action='store', metavar='NAME', type=str, default=None,
                    help='Files with the outputs to plot, if PATH is given, file is supposed to be '
                         'in PATH directory: [PATH]/[NAME], default: [PATH]/[NAMESPACE]_outputs_{}.tsv')
parser.add_argument('--param', action='store', metavar='NAME', type=str, default=None,
                    help='File with parameters of the network, from which outputs should be plotted, '
                         'if PATH is given, file is supposed to be in PATH directory: [PATH]/[NAME], '
                         'default: [PATH]/[NAMESPACE]_params.txt')
parser.add_argument('--namespace', action='store', metavar='NAME', type=str, default=None,
                    help='Namespace of the analysis, default: established based on [FILE]')
parser.add_argument('-p', '--path', action='store', metavar='DIR', type=str, default=None,
                    help='Working directory.')
parser.add_argument('-o', '--output', action='store', metavar='DIR', type=str, default=None,
                    help='Output directory, default: [PATH]/results/[NAMESPACE]')
parser.add_argument('--seed', action='store', metavar='NUMBER', type=int, default='0',
                    help='Set random seed, default: 0')
args = parser.parse_args()


for file in files:
    matrix = np.load(file)
    ax.boxplot()
