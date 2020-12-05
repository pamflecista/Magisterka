import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


#function for ploting train results
def plot_all_result(run,namespace='custom',trainset=True):

    Dane = [[],[],[],[],[],[]]
    file='{}_train_results.tsv'.format(namespace+str(run))
    run_catalog = namespace + str(run)

    file = Path.cwd().parents[0] / 'results' / run_catalog / file

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



#function for ploting train results starting from given epoch
def plot_after_some_epoch(run,namespace='custom', epoch=150,trainset=True):

    Dane = [[],[],[],[],[],[]]
    file = '{}_train_results.tsv'.format(namespace + str(run))
    run_catalog = namespace + str(run)

    file = Path.cwd().parents[0] / 'results' / run_catalog / file


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


#function for calculating mean and sd for each class, from training results, starting from given epoch

def calculate_mean_and_sd(run,namespace='custom', epoch=150, trainset=True):
    Dane = [[], [], [], [], [], []]
    file = '{}_train_results.tsv'.format(namespace + str(run))
    run_catalog = namespace + str(run)

    file = Path.cwd().parents[0] / 'results' / run_catalog / file


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
            f.write("Run    Stage      Dropout  Momentum    lr  Mean of  PA  NPA PIN NPIN    Sd of   PA  NPA PIN NPIN")

    file_name='{}_pamfl_params.csv'.format(namespace+str(run))

    run_catalog=namespace + str(run)

    path_to_file=Path.cwd().parents[0] / 'results'/ run_catalog / file_name
    with open(path_to_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader, None)
        for line in reader:
            dropout_val=float(line[0])
            momentum_val=float(line[1])
            lr=float(line[2])
    if train:
        stage='train'
    else:
        stage='valid'
    mas=calculate_mean_and_sd(run, namespace='custom', epoch=epoch, trainset=train)

    with open(my_file, 'a') as f:

        str_to_write = '{}    {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}'.format(
            run, stage, dropout_val, momentum_val, lr, mas['pa'][0], mas['npa'][0],
            mas['pin'][0], mas['npin'][0], mas['pa'][1], mas['npa'][1],
            mas['pin'][1], mas['npin'][1]
        )
        f.write(str_to_write)


for i in range(50,35,-1):
    pamfl_write_result(i,epoch=150,namespace='custom', train=False)


def pamfl_mean_and_sd_of_many_runs(run_start,run_end,epoch=150,namespace='custom', train=False):
    my_file = Path("pamfl_results.tsv")
    if train:
        stage = 'train'
    else:
        stage = 'valid'
    f=open(my_file, 'w')
    f.close()
    results={}
    for run in range(run_start,run_end+1):
        pamfl_write_result(run, epoch, namespace, train)
    with open(my_file, 'r') as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader, None)
        for line in reader:

            dropout=line[2]
            momentum=line[3]
            lr=line[4]
            mean_pa=float(line[5])
            mean_npa = float(line[6])
            mean_pin = float(line[7])
            mean_npin = float(line[8])

            sd_pa=float(line[9])
            sd_npa = float(line[10])
            sd_pin = float(line[11])
            sd_npin = float(line[12])

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
                results[key]['mean_npin'] = sum(results[key]['mean_pin']) / length

                results[key]['sd_pa'] = sum(results[key]['sd_pa']) / length
                results[key]['sd_npa'] = sum(results[key]['sd_npa']) / length
                results[key]['sd_pin'] = sum(results[key]['sd_pin']) / length
                results[key]['sd_npin'] = sum(results[key]['sd_npin']) / length

            my_file=Path("pamfl_mean_results.tsv")

            with open(my_file, 'w') as f:
                f.write(
                    "Stage      Dropout  Momentum    lr  Mean of  PA  NPA PIN NPIN    Mean_Sd of   PA  NPA PIN NPIN     Sd of   PA  NPA PIN NPIN")
                for key in results.keys():
                    str_to_write = '{}    {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}  {}    {}  {}  {}'.format(
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







