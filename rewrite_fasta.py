import argparse
import os


def rewrite_fasta(file, outdir=None, name_pos=None):
    with open(file, 'r') as f:
        i = 0
        line = f.readline()
        while i < 2 and line:
            if line.startswith('>'):
                i += 1
            line = f.readline()
        if i == 1:
            print('No rewriting was done: given file contains only one sequence.')
            return 1, ['/'+'/'.join(file.split('/')[:-1])]
    if outdir is None:
        outdir, name = os.path.split(file)
        namespace, _ = os.path.splitext(name)
        outdir = os.path.join(outdir, namespace)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
    i = 0
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                if name_pos is not None:
                    filename = line.split(' ')[name_pos] + '.fasta'
                else:
                    filename = ':'.join(line.split(' ')[1:3]).strip('chr ') + '.fasta'
                w = open(os.path.join(outdir, filename), 'w')
                w.write(line)
                i += 1
            else:
                w.write(line)
                w.close()
    print('Based on {} {} sequences were written into separated files in {}'.format(file, i, outdir))
    return i, outdir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rewrite sequences to separated files')
    parser.add_argument('file', action='store', metavar='FILE', type=str,
                        help='Fasta file or directory with files to rewrite')
    parser.add_argument('-e', '--extension', action='store', metavar='EXT', type=str, default='fa',
                        help="Extension of the files in the given [PATH] which should be rewritten, default 'fa'")
    parser.add_argument('-o', '--output', action='store', metavar='DIR', type=str, default=None,
                        help='Directory into which the sequences should be rewritten, by default it is directory of the'
                             ' input file in which new folder is created.')
    args = parser.parse_args()

    if os.path.isdir(args.file):
        for f in [el for el in os.listdir(args.file) if os.path.isfile(os.path.join(args.file, el)) and
                  el.endswith(args.extension)]:
            rewrite_fasta(os.path.join(args.file, f), args.output)
    else:
        rewrite_fasta(args.file, args.output)
