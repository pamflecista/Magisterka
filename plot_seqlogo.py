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


def create_counts(arr, max_v=None, min_v=None):
    mot_len = arr.shape[1]
    res_pos = []
    res_neg = []
    if max_v is None:
        max_v = np.max(arr)
    if min_v is None:
        min_v = np.min(arr)
    for pos in range(mot_len):
        val_pos = []
        val_neg = []
        for (nuc, v) in zip("ACGT", arr[:, pos]):
            if v > 0:
                val_pos.append((nuc, v / max_v))
                val_neg.append((nuc, 0.0))
            else:
                val_pos.append((nuc, 0.0))
                val_neg.append((nuc, v / min_v))
        res_pos.append(val_pos)
        res_neg.append(val_neg)
    return res_pos, res_neg


def create_pos_neg_plots(arr, pos, step=20, prefix=""):
    working_dir = os.path.join(output, 'seqlogo')
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)
    mav = np.max(arr)
    miv = np.min(arr)
    res_pos, res_neg = create_counts(arr[:, pos:pos + step], mav, miv)
    f_pos, ax_pos = pyseqlogo.draw_logo(res_pos, data_type="bits", draw_axis=1)
    pylab.xticks(range(1, step + 1), [""] * step)
    pylab.yticks([0, 1, 2], ["0", "", str(2 * mav)])
    pylab.xlabel(str(pos + step / 2))
    f_pos.savefig(os.path.join(working_dir, prefix + "_pos_%d.png" % pos))
    pylab.close()
    f_neg, ax_neg = pyseqlogo.draw_logo(res_neg, data_type="bits", draw_axis=1)
    pylab.xticks(range(1, step + 1), [""] * step)
    pylab.yticks([0, 1, 2], ["0", "", str(2 * miv)])
    f_neg.savefig(os.path.join(working_dir, prefix + "_neg_%d.png" % pos))
    pylab.close()
    return working_dir


def stitch(arr, name_prefix="stitched", step=20, aspect=5):
    n = Image.new(mode="RGB", size=(int(1640 * arr.shape[1] / step), 600))
    for position in range(0, arr.shape[1], step):
        print(name_prefix, position)
        working_dir = create_pos_neg_plots(arr, position, step=step, prefix=name_prefix)
        img_pos = Image.open(os.path.join(working_dir, name_prefix + "_pos_%d.png" % position))
        img_neg = Image.open(os.path.join(working_dir, name_prefix + "_neg_%d.png" % position))
        n.paste(img_neg.crop((300, 0, 1940, 300)).transpose(Image.FLIP_TOP_BOTTOM), (int(position / step * 1640), 300))
        n.paste(img_pos.crop((300, 0, 1940, 300)), (int(position / step * 1640), 0))
    n.resize((int(n.size[0] / aspect), n.size[1]), Image.ANTIALIAS).save(os.path.join(output, name_prefix + ".png"))


for seq, label, d in zip(integrads, labels, desc):
    name = '{}:{}'.format(classes[label].replace(' ', '-'), d)
    print('Plotting {}'.format(name))
    stitch(seq[:, 850:1150], name_prefix=name)
