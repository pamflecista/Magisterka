import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_all_result(file,trainset=True):

    Dane = [[],[],[],[],[],[]]

    with open(file) as tsvfile:
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
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value of Cross-entropy loss')
    blue_patch = mpatches.Patch(color='blue', label='promoter active')
    green_patch = mpatches.Patch(color='green', label='nonpromoter active')
    red_patch = mpatches.Patch(color='red', label='promoter inactive')
    yellow_patch = mpatches.Patch(color='yellow', label='nonpromoter inactive')
    plt.legend(handles=[red_patch,blue_patch,green_patch,yellow_patch])
    plt.show()


#plot_all_result('custom35_train_results.tsv',True)
#plot_all_result('custom35_train_results.tsv',False)

def plot_after_some_epoch(file, epoch,trainset=True):

    Dane = [[],[],[],[],[],[]]

    with open(file) as tsvfile:
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
    Dane=Dane[:,epoch*2:]

    a=Dane[1]=='train'
    train=Dane[:,a]
    valid=Dane[:,~a]

    train_bool=trainset

    if train_bool:
        dane=train
    else:
        dane=valid

    fig, ax =plt.subplots()

    plt.setp(ax,xticks=range(0,200-epoch,10))
    ax.plot(dane[0],list(map(float,dane[2])),marker='o', c='blue', ms=2, lw=0)
    ax.plot(dane[0],list(map(float,dane[3])),marker='o', c='green', ms=2, lw=0)
    ax.plot(dane[0],list(map(float,dane[4])),marker='o', c='red', ms=2, lw=0)
    ax.plot(dane[0],list(map(float,dane[5])),marker='o', c='yellow', ms=2, lw=0 )
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value of Cross-entropy loss')
    blue_patch = mpatches.Patch(color='blue', label='promoter active')
    green_patch = mpatches.Patch(color='green', label='nonpromoter active')
    red_patch = mpatches.Patch(color='red', label='promoter inactive')
    yellow_patch = mpatches.Patch(color='yellow', label='nonpromoter inactive')
    plt.legend(handles=[red_patch,blue_patch,green_patch,yellow_patch])

    plt.show()

plot_after_some_epoch('custom35_train_results.tsv',150,True)
plot_after_some_epoch('custom35_train_results.tsv',150,False)


def calculate_mean_and_sd(file, epoch, trainset=True):
    Dane = [[], [], [], [], [], []]

    with open(file) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader, None)
        for line in tsvreader:
            a, b, c, d = line[2].split(' ')
            Dane[0].append(line[0])
            Dane[1].append(line[1])
            Dane[2].append(a.strip(','))
            Dane[3].append(b.strip(','))
            Dane[4].append(c.strip(','))
            Dane[5].append(d.strip(','))

    Dane = np.array(Dane)
    Dane = Dane[:, epoch * 2:]

    a = Dane[1] == 'train'
    train = Dane[:, a]
    valid = Dane[:, ~a]

    train_bool = trainset

    if train_bool:
        dane = train
    else:
        dane = valid

    mean_of_promoter_active=sum(list(map(float, dane[2])))/len(dane[2])
    mean_of_nonpromoter_active = sum(list(map(float, dane[3]))) / len(dane[3])
    mean_of_promoter_inactive = sum(list(map(float, dane[4]))) / len(dane[4])
    mean_of_nonpromoter_inactive = sum(list(map(float, dane[5]))) / len(dane[5])

    sd_of_promoter_active = np.std(list(map(float, dane[2])))
    sd_of_nonpromoter_active = np.std(list(map(float, dane[3])))
    sd_of_promoter_inactive = np.std(list(map(float, dane[4])))
    sd_of_nonpromoter_inactive = np.std(list(map(float, dane[5])))

    result={"pa":(mean_of_promoter_active,sd_of_promoter_active),"npa":(mean_of_nonpromoter_active
                                                                        ,sd_of_nonpromoter_active),
            "pin":(mean_of_promoter_inactive,sd_of_promoter_inactive),
            "npin":(mean_of_nonpromoter_inactive,sd_of_nonpromoter_inactive)}

    return result



print(calculate_mean_and_sd('custom35_train_results.tsv',150,True)["pa"])
print(calculate_mean_and_sd('custom35_train_results.tsv',150,False)["pa"])