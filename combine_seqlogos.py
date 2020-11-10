import argparse
import os, sys
from bin.common import *
import warnings
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

parser = argparse.ArgumentParser(description='Combine seqlogos from the given directories')
parser.add_argument('dir1', action='store', metavar='DIR', type=str, default=None,
                    help='First directory containing folders with seqlogos (seqlogo, seqlogo-global-ylim, '
                         'seqlogo-one, seqlogo-one-global-ylim)')
parser.add_argument('dir2', action='store', metavar='DIR', type=str, default=None,
                    help='Second directory containing folders with seqlogos (seqlogo, seqlogo-global-ylim, '
                         'seqlogo-one, seqlogo-one-global-ylim)')
parser.add_argument('--clip', action='store', metavar='VALUE', type=str, default=None,
                    help='Clip parameter from plotting seqlogos')
parser.add_argument('--name', action='store', metavar='NAME', type=str, default=None,
                    help='Name of the output directory')
parser = basic_params(parser, param=True)
args = parser.parse_args()
path, output, namespace, seed = parse_arguments(args, args.dir1)

if not os.path.isdir(args.dir1):
    dir1 = os.path.join(path, args.dir1)
else:
    dir1 = args.dir1
if not os.path.isdir(args.dir2):
    dir2 = os.path.join(path, args.dir2)
else:
    dir2 = args.dir2

if args.clip is not None:
    clip = '-{}'.format(args.clip)
else:
    clip = ''

if args.name is None:
    _, name1 = os.path.split(dir1)
    name1 = name1.split('_')[-2:]
    name1 = '{}:{}'.format(name1[0].replace('1', ''), name1[1])
    _, name2 = os.path.split(dir2)
    name2 = name2.split('_')[-2:]
    name2 = '{}:{}'.format(name2[0].replace('2', ''), name2[1])
    assert name1 == name2
    name = name1
else:
    name = args.name
outdir = os.path.join(output, name)
if os.path.isdir(outdir):
    warnings.warn('\nAnalysis in {} already exists, it will be overwritten'.format(outdir))
    import shutil
    shutil.rmtree(outdir)
os.mkdir(outdir)

for type_ in ['seqlogo{}'.format(clip), 'seqlogo{}-global-ylim'.format(clip), 'seqlogo{}-one'.format(clip),
              'seqlogo{}-one-global-ylim'.format(clip), 'seqlogo{}-average'.format(clip)]:
    for class_ in ['promoter-active', 'promoter-inactive', 'nonpromoter-active', 'nonpromoter-inactive']:
        for extreme_ in ['best', 'worst']:
            namespace = '{} : {} - {} - {}'.format(name, type_, class_, extreme_)
            try:
                if 'average' in type_:
                    img1 = Image.open(os.path.join(dir1, type_, '{}:average-{}.png'.format(class_, extreme_)))
                    img2 = Image.open(os.path.join(dir2, type_, '{}:average-{}.png'.format(class_, extreme_)))
                else:
                    img1 = Image.open(os.path.join(dir1, type_, '{}:{}.png'.format(class_, extreme_)))
                    img2 = Image.open(os.path.join(dir2, type_, '{}:{}.png'.format(class_, extreme_)))
                print(namespace)
            except FileNotFoundError:
                print('Images for {} do not exist!'.format(namespace))
                print(os.path.join(dir1, type_, '{}:average-{}.png'.format(class_, extreme_)))
                continue
            new = Image.new(mode="RGB", size=(max(img1.size[0], img2.size[0])+20, img1.size[1]+img2.size[1]+400), color=(255,255,255,0))
            new.paste(img1, (10, 200))
            new.paste(img2, (10, img1.size[1]+400))
            draw = ImageDraw.Draw(new)
            font = ImageFont.truetype("/home/marni/magisterka/arial.ttf", 35)
            # draw.text((x, y),"Sample Text",(r,g,b))
            draw.text((new.size[0]/2-500, 50), namespace + '- #1', (0, 0, 0), font=font)
            draw.text((new.size[0]/2-500, img1.size[1]+200), namespace + '- #2', (0, 0, 0), font=font)
            new.save(os.path.join(outdir, '{}_{}_{}.png'.format(type_, class_, extreme_)))
