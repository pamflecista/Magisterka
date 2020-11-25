import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_result(plik,trainset=True):

    Dane = [[],[],[],[],[],[]]
    i=0
    with open(plik) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)
        for line in tsvreader:
            a,b,c,d=line[2].split(' ')
            Dane[0].append(line[0])
            Dane[1].append(line[1])
            Dane[2].append(a.strip(','))
            Dane[3].append(b.strip(','))
            Dane[4].append(c.strip(','))
            Dane[5].append(d.strip(','))

    Dane=np.array(Dane)
    a=Dane[1]=='train'
    train=Dane[:,a]
    valid=Dane[:,~a]

    train_bool=trainset

    if train_bool:
        dane=train
    else:
        dane=valid

    fig, ax =plt.subplots()
    ax.set_xticks([24,49,74,99,124,149,174,199,224,249,274,299,324,349,374,399])

    ax.plot(dane[0],list(map(float,dane[2])),marker='o', c='blue', ms=2, lw=0)
    ax.plot(dane[0],list(map(float,dane[3])),marker='o', c='green', ms=2, lw=0)
    ax.plot(dane[0],list(map(float,dane[4])),marker='o', c='red', ms=2, lw=0)
    ax.plot(dane[0],list(map(float,dane[5])),marker='o', c='yellow', ms=2, lw=0 )
    blue_patch = mpatches.Patch(color='blue', label='promoter active')
    green_patch = mpatches.Patch(color='green', label='nonpromoter active')
    red_patch = mpatches.Patch(color='red', label='promoter inactive')
    yellow_patch = mpatches.Patch(color='yellow', label='nonpromoter inactive')
    plt.legend(handles=[red_patch,blue_patch,green_patch,yellow_patch])
    plt.show()


plot_result('custom35_train_results.tsv',True)
plot_result('custom35_train_results.tsv',False)

