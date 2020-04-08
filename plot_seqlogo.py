import pyseqlogo
import numpy as np
from matplotlib import pylab
from PIL import Image
import argparse
import os
from bin.common import *

parser = argparse.ArgumentParser(description='Plot sequence logo for given integrads matrix')
parser.add_argument('integrads', action='store', metavar='FILE', type=str, default=None,
                    help='Numpy array with integrads (output of calculate_integrads.py script)')
parser.add_argument('--global_ylim', action='store_true',
                    help='If ylim should be set global for all classes')
parser.add_argument('--one', action='store_true',
                    help='If only one nucleotide (from the origin sequence) should be plot')
parser = basic_params(parser, param=True)
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, args.integrads)

integrads = np.load(args.integrads)
with open(os.path.join(path, 'params.txt'), 'r') as f:
    for line in f:
        if line.startswith('Seq labels'):
            labels = list(map(int, line.strip().split(': ')[1].split(', ')))
        elif line.startswith('Seq descriptions'):
            desc = line.strip().split(': ')[1].split(', ')
        elif line.startswith('Classes'):
            classes = line.strip().split(': ')[1].split(', ')
        elif line.startswith('Seq file'):
            seq_file = line.strip().split(': ')[1]
        elif line.startswith('Seq IDs'):
            seq_ids = line.strip().split(': ')[1].split(', ')

seqs = {}
with open(seq_file, 'r') as f:
    for line in f:
        if line.startswith('>'):
            l = line.strip().split(' ')
            seq_id = '{}:{}'.format(l[1].lstrip('chr'), l[2])
        else:
            seqs[seq_id] = line.strip()
seqs = [seqs[el] for el in seq_ids]


def create_counts(arr, seq, max_v=None, min_v=None, one=False):
    mot_len = arr.shape[1]
    assert mot_len == len(seq)
    res_pos = []
    res_neg = []
    if max_v is None:
        max_v = np.max(arr)
    if min_v is None:
        min_v = np.min(arr)
    for pos, letter in enumerate(seq):
        val_pos = []
        val_neg = []
        for (nuc, v) in zip("ACGT", arr[:, pos]):
            if nuc == letter or not one:
                if v > 0:
                    val_pos.append((nuc, v / max_v))
                    val_neg.append((nuc, 0.0))
                else:
                    val_pos.append((nuc, 0.0))
                    val_neg.append((nuc, v / min_v))
            else:
                val_pos.append((nuc, 0.0))
                val_neg.append((nuc, 0.0))
        res_pos.append(val_pos)
        res_neg.append(val_neg)
    return res_pos, res_neg


def create_pos_neg_plots(arr, seq, pos, outdir, step=20, prefix="", ylim=None, one=False):
    if ylim is None:
        ylim = np.max(np.absolute(arr))
    res_pos, res_neg = create_counts(arr[:, pos:pos + step], seq[pos:pos+step], max_v=ylim, min_v=-ylim, one=one)
    f_pos, ax_pos = pyseqlogo.draw_logo(res_pos, data_type="bits", draw_axis=1)
    pylab.xticks(range(1, step + 1), [""] * step)
    pylab.yticks([0, 1, 2], ["0", "", str(2 * ylim)])
    pylab.xlabel(str(pos + step / 2))
    f_pos.savefig(os.path.join(outdir, prefix + "_pos_%d.png" % pos))
    pylab.close()
    f_neg, ax_neg = pyseqlogo.draw_logo(res_neg, data_type="bits", draw_axis=1)
    pylab.xticks(range(1, step + 1), [""] * step)
    pylab.yticks([0, 1, 2], ["0", "", str(2 * -ylim)])
    f_neg.savefig(os.path.join(outdir, prefix + "_neg_%d.png" % pos))
    pylab.close()


def stitch(arr, seq, outdir, name_prefix="stitched", step=20, aspect=5, ylim=None, one=False):
    n = Image.new(mode="RGB", size=(int(1640 * arr.shape[1] / step), 600))
    for position in range(0, arr.shape[1], step):
        print(name_prefix, position)
        if ylim is None:
            ylim = np.max(np.absolute(arr))
            print('YLIM: {}'.format(ylim))
        create_pos_neg_plots(arr, seq, position, os.path.join(outdir, 'subplots'), step=step, prefix=name_prefix, ylim=ylim, one=one)
        img_pos = Image.open(os.path.join(outdir, 'subplots', name_prefix + "_pos_%d.png" % position))
        img_neg = Image.open(os.path.join(outdir, 'subplots', name_prefix + "_neg_%d.png" % position))
        n.paste(img_neg.crop((300, 0, 1940, 300)).transpose(Image.FLIP_TOP_BOTTOM), (int(position / step * 1640), 300))
        n.paste(img_pos.crop((300, 0, 1940, 300)), (int(position / step * 1640), 0))
    n.resize((int(n.size[0] / aspect), n.size[1]), Image.ANTIALIAS).save(os.path.join(outdir, name_prefix + ".png"))


one = False
global_ylim = None
clip = 100
name = 'seqlogo-{}'.format(clip)
if args.one:
    name += '-one'
    one = True
if args.global_ylim:
    global_ylim = np.max(np.absolute(integrads[:, :, 1000-clip:1000+clip]))
    print('Global ylim set to {}'.format(global_ylim))
    name += '-global-ylim'
working_dir = os.path.join(output, name)
if not os.path.isdir(working_dir):
    os.mkdir(working_dir)
if not os.path.isdir(os.path.join(working_dir, 'subplots')):
    os.mkdir(os.path.join(working_dir, 'subplots'))
for seq, grads, label, d in zip(seqs, integrads, labels, desc):
    name = '{}:{}'.format(classes[label].replace(' ', '-'), d)
    print('Plotting {}'.format(name))
    stitch(grads[:, 1000-clip:1000+clip], seq, working_dir, name_prefix=name, ylim=global_ylim, one=one)
