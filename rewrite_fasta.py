import argparse
import os

parser = argparse.ArgumentParser(description='Rewrite sequences to separated files')
parser.add_argument('file', action='store', metavar='FILE', type=str,
                    help='Fasta file to rewrite')
parser.add_argument('-o', '--output', action='store', metavar='DIR', type=str,
                    help='Directory into which the sequences should be rewritten, by default it is directory of the '
                         'input file in which new folder is created.')
args = parser.parse_args()


def rewrite_fasta(file, path=None):
    if path is None:
        f = file.split('/')
        path = os.path.join('/', *f[:-1], '.'.join(f[-1].split('.')[:-1]))
        if not os.path.isdir(path):
            os.mkdir(path)
    i = 0
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                filename = ':'.join(line.split(' ')[1:3]).strip('chr ') + '.fasta'
                w = open(os.path.join(path, filename), 'w')
                w.write(line)
                i += 1
            else:
                w.write(line)
                w.close()
    print('Based on {} {} sequences were written into separated files in {}'.format(file, i, path))


rewrite_fasta(args.file, args.output)
