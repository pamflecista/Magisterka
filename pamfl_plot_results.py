import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import os


#function for ploting train results
def plot_all_result(run,namespace='custom',trainset=True):

    Dane = [[],[],[],[],[],[]]
    file='{}_train_results.tsv'.format(namespace+str(run))
    run_catalog = namespace + str(run)

    file = Path.cwd().parents[0] / 'results' / run_catalog / file

    file_name = '{}_pamfl_params.csv'.format(namespace + str(run))

    path_to_file = Path.cwd().parents[0] / 'results' / run_catalog / file_name


    dir_path = Path.cwd().parents[0] / 'results' / run_catalog

    if os.path.isdir(dir_path):
        with open(path_to_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader, None)
            for line in reader:
                dropout_val = float(line[0])
                momentum_val = float(line[1])
                lr = float(line[2])
                conv_dropout=float(line[3])

        if trainset:
            stage = 'train'
        else:
            stage = 'valid'

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
        fig.suptitle(
            'stage: {}, dropout={}, momentum={}, learning rate={}, conv dropout={}'.format(stage, dropout_val, momentum_val, lr, conv_dropout),
            fontsize=9, fontweight='bold')

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
    else:
        print('no such directory {}'.format(dir_path))


#plot_all_result(36,namespace='custom',trainset=True)

#function for ploting train results starting from given epoch
def plot_after_some_epoch(run,namespace='custom', epoch=150,trainset=True):

    Dane = [[],[],[],[],[],[]]
    file = '{}_train_results.tsv'.format(namespace + str(run))
    run_catalog = namespace + str(run)

    file = Path.cwd().parents[0] / 'results' / run_catalog / file

    file_name = '{}_pamfl_params.csv'.format(namespace + str(run))



    path_to_file = Path.cwd().parents[0] / 'results' / run_catalog / file_name
    dir_path = Path.cwd().parents[0] / 'results' / run_catalog
    if os.path.isdir(dir_path):
        with open(path_to_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader, None)
            for line in reader:
                dropout_val = float(line[0])
                momentum_val = float(line[1])
                lr = float(line[2])
                conv_dropout=float(line[3])
        if trainset:
            stage = 'train'
        else:
            stage = 'valid'


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

        fig.suptitle('stage: {}, dropout={}, momentum={}, learning rate={}, convolutional dropout={}'.format(stage, dropout_val, momentum_val, lr,conv_dropout), fontsize=9,
                     fontweight='bold')

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
    else:
        print('no such directory {}'.format(dir_path))

#plot_after_some_epoch(30,namespace='custom', epoch=150,trainset=True)


#function for calculating mean and sd for each class, from training results, starting from given epoch

def calculate_mean_and_sd(run,namespace='custom', epoch=150, trainset=True):
    Dane = [[], [], [], [], [], []]
    file = '{}_train_results.tsv'.format(namespace + str(run))
    run_catalog = namespace + str(run)

    file = Path.cwd().parents[0] / 'results' / run_catalog / file
    dir=Path.cwd().parents[0] / 'results' / run_catalog

    if os.path.isdir(dir):

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

#function for writing calculated mean and sd for each class to {}_pamfl_results.tsv file


def pamfl_write_result(run,epoch=150,namespace='custom', train=False):
    my_file = Path("pamfl_results.tsv")
    if not my_file.is_file():
        with open(my_file,'w') as f:
            f.write("Run    Stage      Dropout  Momentum    lr  Conv_Dropout  Mean of  PA  NPA PIN NPIN    Sd of   PA  NPA PIN NPIN\n")

    file_name='{}_pamfl_params.csv'.format(namespace+str(run))

    run_catalog=namespace + str(run)

    path_to_file=Path.cwd().parents[0] / 'results'/ run_catalog / file_name
    dir_path=Path.cwd().parents[0] / 'results'/ run_catalog
    if os.path.isdir(dir_path):
        with open(path_to_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader, None)
            for line in reader:
                dropout_val=float(line[0])
                momentum_val=float(line[1])
                lr=float(line[2])
                conv_dropout=float(line[3])
        if train:
            stage='train'
        else:
            stage='valid'
        mas=calculate_mean_and_sd(run, namespace='custom', epoch=epoch, trainset=train)

        with open(my_file, 'a') as f:

            str_to_write = '{}    {}  {}    {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}\n'.format(
                run, stage, dropout_val, momentum_val, lr, conv_dropout, mas['pa'][0], mas['npa'][0],
                mas['pin'][0], mas['npin'][0], mas['pa'][1], mas['npa'][1],
                mas['pin'][1], mas['npin'][1]
            )
            f.write(str_to_write)


#for i in range(75,35,-1):
   #pamfl_write_result(i,epoch=150,namespace='custom', train=False)

#function which creates file which consists of means of many runs for each value of dropout

def pamfl_mean_and_sd_of_many_runs(run_start,run_end,epoch=150,namespace='custom', train=False,
                                   cdrop=False):
    my_file = Path("pamfl_results.tsv")
    if train:
        stage = 'train'
    else:
        stage = 'valid'

    results={}
    for run in range(run_start,run_end+1):
        pamfl_write_result(run, epoch, namespace, train)
    with open(my_file, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader, None)
        for line in reader:
            line=line[0].split()
            if cdrop:
                dropout=line[5]
                kind_of_dropout='conv-dropout'
            else:
                kind_of_dropout='dropout'
                dropout=line[2]
            momentum=line[3]
            lr=line[4]
            mean_pa=float(line[6])
            mean_npa = float(line[7])
            mean_pin = float(line[8])
            mean_npin = float(line[9])

            sd_pa=float(line[10])
            sd_npa = float(line[11])
            sd_pin = float(line[12])
            sd_npin = float(line[13])

            if dropout not in results:
                results[dropout]={'params':[dropout,momentum,lr],'mean_pa':[mean_pa],
                                  'mean_npa':[mean_npa],'mean_pin':[mean_pin],
                                  'mean_npin':[mean_npin],'sd_pa':[sd_pa],'sd_npa':[sd_npa],
                                  'sd_pin':[sd_pin],'sd_npin':[sd_npin]}
            else:
                results[dropout]['mean_pa'].append(mean_pa)
                results[dropout]['mean_npa'].append(mean_npa)
                results[dropout]['mean_pin'].append(mean_pin)
                results[dropout]['mean_npin'].append(mean_npin)

                results[dropout]['sd_pa'].append(sd_pa)
                results[dropout]['sd_npa'].append(sd_npa)
                results[dropout]['sd_pin'].append(sd_pin)
                results[dropout]['sd_npin'].append(sd_npin)







        for key in results.keys():

            results[key]['sd_of_mean_pa']=np.std(results[key]['mean_pa'])
            results[key]['sd_of_mean_npa'] = np.std(results[key]['mean_npa'])
            results[key]['sd_of_mean_pin'] = np.std(results[key]['mean_pin'])
            results[key]['sd_of_mean_npin'] = np.std(results[key]['mean_npin'])
            length=len(results[key]['mean_pa'])


            results[key]['mean_pa']=sum(results[key]['mean_pa'])/length
            results[key]['mean_npa'] = sum(results[key]['mean_npa']) / length
            results[key]['mean_pin'] = sum(results[key]['mean_pin']) / length
            results[key]['mean_npin'] = sum(results[key]['mean_npin']) / length

            results[key]['sd_pa'] = sum(results[key]['sd_pa']) / length
            results[key]['sd_npa'] = sum(results[key]['sd_npa']) / length
            results[key]['sd_pin'] = sum(results[key]['sd_pin']) / length
            results[key]['sd_npin'] = sum(results[key]['sd_npin']) / length

        my_file=Path("pamfl_mean_results.tsv")

        with open(my_file, 'w') as f:
            f.write(
                "Stage      {}  Momentum    lr  Mean of  PA  NPA PIN NPIN    Mean_Sd of   PA  NPA PIN NPIN    "
                " Sd of   PA  NPA PIN NPIN\n".format(kind_of_dropout))
            for key in results.keys():
                str_to_write = '{}    {}  {}  {}        {:.4f}  {:.4f}  {:.4f}  {:.4f}       {:.4f}  {:.4f}  {:.4f}  {:.4f}          {:.4f}    {:.4f}  {:.4f}  {:.4f}\n'.format(
                stage, results[key]['params'][0], results[key]['params'][1], results[key]['params'][2],
                results[key]['mean_pa'],
                results[key]['mean_npa'],
                results[key]['mean_pin'],
                results[key]['mean_npin'],

                results[key]['sd_of_mean_pa'],
                results[key]['sd_of_mean_npa'],
                results[key]['sd_of_mean_pin'],
                results[key]['sd_of_mean_npin'],

                results[key]['sd_pa'],
                results[key]['sd_npa'],
                results[key]['sd_pin'],
                results[key]['sd_npin'],




                )
                f.write(str_to_write)



#pamfl_mean_and_sd_of_many_runs(36,61,epoch=175,namespace='custom', train=False)

#function  for plotting loss vs dropout

def pamfl_plot_mean_vs_dropout(file):
    dropout=[]
    momentum=set()
    lr=set()
    stage=set()

    mean_pa=[]
    mean_npa=[]
    mean_pin = []
    mean_npin = []

    sd_of_means_pa=[]
    sd_of_means_npa = []
    sd_of_means_pin = []
    sd_of_means_npin = []

    mean_of_sd_pa=[]
    mean_of_sd_npa = []
    mean_of_sd_pin = []
    mean_of_sd_npin = []

    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for line in reader:
            line=line[0].split()
            dropout.append(line[1])
            momentum.add(line[2])
            lr.add(line[3])
            stage.add(line[0])

            mean_pa.append(line[4])
            mean_npa.append(line[5])
            mean_pin.append(line[6])
            mean_npin.append(line[7])

            sd_of_means_pa.append(line[8])
            sd_of_means_npa.append(line[9])
            sd_of_means_pin.append(line[10])
            sd_of_means_npin.append(line[11])

            mean_of_sd_pa.append(line[12])
            mean_of_sd_npa.append(line[13])
            mean_of_sd_pin.append(line[14])
            mean_of_sd_npin.append(line[15])

    sort_index=sorted(range(len(dropout)), key=lambda k: dropout[k])



    dropout.sort()

    sort_index= [x for _, x in sorted(zip(sort_index, range(len(sort_index))))]



    mean_pa=list(map(float, mean_pa))
    mean_pa = [x for _, x in sorted(zip(sort_index, mean_pa))]

    mean_npa = list(map(float, mean_npa))
    mean_npa = [x for _, x in sorted(zip(sort_index, mean_npa))]
    mean_pin = list(map(float, mean_pin))
    mean_pin = [x for _, x in sorted(zip(sort_index, mean_pin))]
    mean_npin = list(map(float, mean_npin))
    mean_npin = [x for _, x in sorted(zip(sort_index, mean_npin))]



    sd_of_means_pa=list(map(float, sd_of_means_pa))
    sd_of_means_pa = [x for _, x in sorted(zip(sort_index, sd_of_means_pa))]
    sd_of_means_npa = list(map(float, sd_of_means_npa))
    sd_of_means_npa = [x for _, x in sorted(zip(sort_index, sd_of_means_npa))]
    sd_of_means_pin = list(map(float, sd_of_means_pin))
    sd_of_means_pin = [x for _, x in sorted(zip(sort_index, sd_of_means_pin))]
    sd_of_means_npin = list(map(float, sd_of_means_npin))
    sd_of_means_npin = [x for _, x in sorted(zip(sort_index, sd_of_means_npin))]

    mean_of_sd_pa = list(map(float, mean_of_sd_pa))
    mean_of_sd_pa = [x for _, x in sorted(zip(sort_index, mean_of_sd_pa))]
    mean_of_sd_npa = list(map(float, mean_of_sd_npa))
    mean_of_sd_npa = [x for _, x in sorted(zip(sort_index, mean_of_sd_npa))]
    mean_of_sd_pin = list(map(float, mean_of_sd_pin))
    mean_of_sd_pin = [x for _, x in sorted(zip(sort_index, mean_of_sd_pin))]
    mean_of_sd_npin = list(map(float, mean_of_sd_npin))
    mean_of_sd_npin = [x for _, x in sorted(zip(sort_index, mean_of_sd_npin))]

    fig, ax = plt.subplots()

    #fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
    ax.plot(dropout, mean_pa, marker='o', c='blue', ms=2, lw=0.1)
    ax.plot(dropout, mean_npa, marker='o', c='green', ms=2, lw=0.1)
    ax.plot(dropout, mean_pin, marker='o', c='red', ms=2, lw=0.1)
    ax.plot(dropout, mean_npin, marker='o', c='black', ms=2, lw=0.1)
    ax.set_xlabel('p dropout')
    ax.set_ylabel('Mean value of Cross-entropy loss')
    blue_patch = mpatches.Patch(color='blue', label='promoter active')
    green_patch = mpatches.Patch(color='green', label='nonpromoter active')

    red_patch = mpatches.Patch(color='red', label='promoter inactive')
    black_patch = mpatches.Patch(color='black', label='nonpromoter inactive')
    plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch],prop={'size': 6})

    plt.show()

    fig, ax = plt.subplots()

    # fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
    ax.plot(dropout, sd_of_means_pa, marker='o', c='blue', ms=2, lw=0.1)
    ax.plot(dropout, sd_of_means_npa, marker='o', c='green', ms=2, lw=0.1)
    ax.plot(dropout, sd_of_means_pin, marker='o', c='red', ms=2, lw=0.1)
    ax.plot(dropout, sd_of_means_npin, marker='o', c='black', ms=2, lw=0.1)
    ax.set_xlabel('p dropout')
    ax.set_ylabel('Standard deviation of mean value of Cross-entropy loss')
    blue_patch = mpatches.Patch(color='blue', label='promoter active')
    green_patch = mpatches.Patch(color='green', label='nonpromoter active')

    red_patch = mpatches.Patch(color='red', label='promoter inactive')
    black_patch = mpatches.Patch(color='black', label='nonpromoter inactive')
    plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch], prop={'size': 6})

    plt.show()

    fig, ax = plt.subplots()

    #fig.suptitle('stage: {} momentum', fontsize=14, fontweight='bold')
    ax.plot(dropout, mean_of_sd_pa, marker='o', c='blue', ms=2, lw=0.1)
    ax.plot(dropout, mean_of_sd_npa, marker='o', c='green', ms=2, lw=0.1)
    ax.plot(dropout, mean_of_sd_pin, marker='o', c='red', ms=2, lw=0.1)
    ax.plot(dropout, mean_of_sd_npin, marker='o', c='black', ms=2, lw=0.1)
    ax.set_xlabel('p dropout')
    ax.set_ylabel('Mean of Standard deviation for trajectory for Cross-entropy loss')
    blue_patch = mpatches.Patch(color='blue', label='promoter active')
    green_patch = mpatches.Patch(color='green', label='nonpromoter active')

    red_patch = mpatches.Patch(color='red', label='promoter inactive')
    black_patch = mpatches.Patch(color='black', label='nonpromoter inactive')
    plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch], prop={'size': 6})

    plt.show()


#pamfl_plot_mean_vs_dropout('pamfl_mean_results.tsv')




def pamfl_get_mean_vs_dropout(file):
    dropout=[]
    momentum=set()
    lr=set()
    stage=set()

    mean_pa=[]
    mean_npa=[]
    mean_pin = []
    mean_npin = []

    sd_of_means_pa=[]
    sd_of_means_npa = []
    sd_of_means_pin = []
    sd_of_means_npin = []

    mean_of_sd_pa=[]
    mean_of_sd_npa = []
    mean_of_sd_pin = []
    mean_of_sd_npin = []

    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for line in reader:
            line=line[0].split()
            dropout.append(line[1])
            momentum.add(line[2])
            lr.add(line[3])
            stage.add(line[0])

            mean_pa.append(line[4])
            mean_npa.append(line[5])
            mean_pin.append(line[6])
            mean_npin.append(line[7])

            sd_of_means_pa.append(line[8])
            sd_of_means_npa.append(line[9])
            sd_of_means_pin.append(line[10])
            sd_of_means_npin.append(line[11])

            mean_of_sd_pa.append(line[12])
            mean_of_sd_npa.append(line[13])
            mean_of_sd_pin.append(line[14])
            mean_of_sd_npin.append(line[15])

    sort_index=sorted(range(len(dropout)), key=lambda k: dropout[k])



    dropout.sort()

    sort_index= [x for _, x in sorted(zip(sort_index, range(len(sort_index))))]



    mean_pa=list(map(float, mean_pa))
    mean_pa = [x for _, x in sorted(zip(sort_index, mean_pa))]

    mean_npa = list(map(float, mean_npa))
    mean_npa = [x for _, x in sorted(zip(sort_index, mean_npa))]
    mean_pin = list(map(float, mean_pin))
    mean_pin = [x for _, x in sorted(zip(sort_index, mean_pin))]
    mean_npin = list(map(float, mean_npin))
    mean_npin = [x for _, x in sorted(zip(sort_index, mean_npin))]



    sd_of_means_pa=list(map(float, sd_of_means_pa))
    sd_of_means_pa = [x for _, x in sorted(zip(sort_index, sd_of_means_pa))]
    sd_of_means_npa = list(map(float, sd_of_means_npa))
    sd_of_means_npa = [x for _, x in sorted(zip(sort_index, sd_of_means_npa))]
    sd_of_means_pin = list(map(float, sd_of_means_pin))
    sd_of_means_pin = [x for _, x in sorted(zip(sort_index, sd_of_means_pin))]
    sd_of_means_npin = list(map(float, sd_of_means_npin))
    sd_of_means_npin = [x for _, x in sorted(zip(sort_index, sd_of_means_npin))]

    mean_of_sd_pa = list(map(float, mean_of_sd_pa))
    mean_of_sd_pa = [x for _, x in sorted(zip(sort_index, mean_of_sd_pa))]
    mean_of_sd_npa = list(map(float, mean_of_sd_npa))
    mean_of_sd_npa = [x for _, x in sorted(zip(sort_index, mean_of_sd_npa))]
    mean_of_sd_pin = list(map(float, mean_of_sd_pin))
    mean_of_sd_pin = [x for _, x in sorted(zip(sort_index, mean_of_sd_pin))]
    mean_of_sd_npin = list(map(float, mean_of_sd_npin))
    mean_of_sd_npin = [x for _, x in sorted(zip(sort_index, mean_of_sd_npin))]

    result={'dropout': dropout, 'means': [mean_pa,mean_npa,mean_pin,mean_npin], 'sd_of_means': [sd_of_means_pa, sd_of_means_npa, sd_of_means_pin, sd_of_means_npin],
            'means_of_sd': [mean_of_sd_pa,mean_of_sd_npa,mean_of_sd_pin,mean_of_sd_npin], 'lr':lr, 'momentum':momentum, 'stage':stage}
    return result

def plot_conv_dropout_vs_no_conv_dropout(crun_start,crun_end,run_start, run_end,cepoch=200,namespace='custom',
                                         train=False,  epoch=150, file='pamfl_mean_results.tsv'):
    my_file = Path("pamfl_results.tsv")
    if my_file.is_file():
        os.remove(my_file)
    pamfl_mean_and_sd_of_many_runs(crun_start, crun_end, epoch=cepoch, namespace=namespace,
                                   train=train, cdrop=True)

    conv_results=pamfl_get_mean_vs_dropout(file)

    dropout=conv_results['dropout']
    mean_pa=conv_results['means'][0]
    mean_npa = conv_results['means'][1]
    mean_pin = conv_results['means'][2]
    mean_npin = conv_results['means'][3]

    sd_of_means_pa=conv_results['sd_of_means'][0]
    sd_of_means_npa = conv_results['sd_of_means'][1]
    sd_of_means_pin = conv_results['sd_of_means'][2]
    sd_of_means_npin = conv_results['sd_of_means'][3]

    mean_of_sd_pa=conv_results['means_of_sd'][0]
    mean_of_sd_npa = conv_results['means_of_sd'][1]
    mean_of_sd_pin = conv_results['means_of_sd'][2]
    mean_of_sd_npin = conv_results['means_of_sd'][3]



    if my_file.is_file():
        os.remove(my_file)
    pamfl_mean_and_sd_of_many_runs(run_start, run_end, epoch=epoch, namespace=namespace,
                                   train=train, cdrop=False)

    non_cov_results=pamfl_get_mean_vs_dropout(file)
    #no convolution dropout

    dropout_noconv=non_cov_results['dropout'][0]
    lr=list(non_cov_results['lr'])[0]
    momentum=list(non_cov_results['momentum'])[0]
    stage=list(non_cov_results['stage'])[0]

    means=non_cov_results['means']
    sd_of_means=non_cov_results['sd_of_means']
    means_of_sd=non_cov_results['means_of_sd']
    print(dropout_noconv)

    fig, ax = plt.subplots()

    fig.suptitle('p dropout={}, lr={}, momentum={}, stage: {}'.format(dropout_noconv,lr,momentum,stage), fontsize=11, fontweight='bold')
    ax.axhline(y=means[0][0], color='blue', linestyle='-')
    ax.plot(dropout, mean_pa, marker='o', c='blue', ms=2, lw=0.1)
    ax.axhline(y=means[1][0], color='green', linestyle='-')
    ax.plot(dropout, mean_npa, marker='o', c='green', ms=2, lw=0.1)
    ax.axhline(y=means[2][0], color='red', linestyle='-')
    ax.plot(dropout, mean_pin, marker='o', c='red', ms=2, lw=0.1)
    ax.axhline(y=means[3][0], color='black', linestyle='-')
    ax.plot(dropout, mean_npin, marker='o', c='black', ms=2, lw=0.1)
    ax.set_xlabel('p convolutional dropout')
    ax.set_ylabel('Mean value of Cross-entropy loss')
    blue_patch = mpatches.Patch(color='blue', label='promoter active')
    green_patch = mpatches.Patch(color='green', label='nonpromoter active')

    red_patch = mpatches.Patch(color='red', label='promoter inactive')
    black_patch = mpatches.Patch(color='black', label='nonpromoter inactive')
    plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch], prop={'size': 6})

    plt.show()

    fig, ax = plt.subplots()

    # fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
    fig.suptitle('p dropout={}, lr={}, momentum={}, stage: {}'.format(dropout_noconv,lr,momentum,stage), fontsize=11, fontweight='bold')
    ax.axhline(y=sd_of_means[0][0], color='blue', linestyle='-')
    ax.plot(dropout, sd_of_means_pa, marker='o', c='blue', ms=2, lw=0.1)
    ax.axhline(y=sd_of_means[1][0], color='green', linestyle='-')
    ax.plot(dropout, sd_of_means_npa, marker='o', c='green', ms=2, lw=0.1)
    ax.axhline(y=sd_of_means[2][0], color='red', linestyle='-')
    ax.plot(dropout, sd_of_means_pin, marker='o', c='red', ms=2, lw=0.1)
    ax.axhline(y=sd_of_means[3][0], color='black', linestyle='-')
    ax.plot(dropout, sd_of_means_npin, marker='o', c='black', ms=2, lw=0.1)
    ax.set_xlabel('p convolutional dropout')
    ax.set_ylabel('Standard deviation of mean value of Cross-entropy loss')
    blue_patch = mpatches.Patch(color='blue', label='promoter active')
    green_patch = mpatches.Patch(color='green', label='nonpromoter active')

    red_patch = mpatches.Patch(color='red', label='promoter inactive')
    black_patch = mpatches.Patch(color='black', label='nonpromoter inactive')
    plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch], prop={'size': 6})

    plt.show()

    fig, ax = plt.subplots()

    # fig.suptitle('stage: {} momentum', fontsize=14, fontweight='bold')
    fig.suptitle('p dropout={}, lr={}, momentum={}, stage: {}'.format(dropout_noconv,lr,momentum,stage), fontsize=11, fontweight='bold')
    ax.axhline(y=means_of_sd[0][0], color='blue', linestyle='-')
    ax.plot(dropout, mean_of_sd_pa, marker='o', c='blue', ms=2, lw=0.1)
    ax.axhline(y=means_of_sd[1][0], color='green', linestyle='-')
    ax.plot(dropout, mean_of_sd_npa, marker='o', c='green', ms=2, lw=0.1)
    ax.axhline(y=means_of_sd[2][0], color='red', linestyle='-')
    ax.plot(dropout, mean_of_sd_pin, marker='o', c='red', ms=2, lw=0.1)
    ax.axhline(y=means_of_sd[3][0], color='black', linestyle='-')
    ax.plot(dropout, mean_of_sd_npin, marker='o', c='black', ms=2, lw=0.1)
    ax.set_xlabel('p convolutional dropout')
    ax.set_ylabel('Mean of Standard deviation for trajectory for Cross-entropy loss')
    blue_patch = mpatches.Patch(color='blue', label='promoter active')
    green_patch = mpatches.Patch(color='green', label='nonpromoter active')

    red_patch = mpatches.Patch(color='red', label='promoter inactive')
    black_patch = mpatches.Patch(color='black', label='nonpromoter inactive')
    plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch], prop={'size': 6})

    plt.show()

plot_conv_dropout_vs_no_conv_dropout(65, 75, 36, 38, cepoch=200, namespace='custom',
                                         train=False, epoch=150)



#for i in range(75,35,-1):
   #pamfl_write_result(i,epoch=150,namespace='custom', train=False)

#pamfl_mean_and_sd_of_many_runs(35,63,epoch=150,namespace='custom', train=False,
                                   #cdrop=False)

#pamfl_mean_and_sd_of_many_runs(64,75,epoch=200,namespace='custom', train=False,
                                   #cdrop=True)





