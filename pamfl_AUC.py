import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import os

def read_data(run,namespace='custom',trainset=True):
    Dane = [[], [],
            [], [], [], [],
            [], [], [], [],
            [], [], [], [],
            [], [], [], [],
            [], [], [], [],
            [], [], [], [],
            [], [], [], []]
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
                conv_dropout = float(line[3])

        if trainset:
            stage = 'train'
        else:
            stage = 'valid'

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
                a, b, c, d = line[3].split(' ')
                Dane[6].append(a.strip(','))
                Dane[7].append(b.strip(','))
                Dane[8].append(c.strip(','))
                Dane[9].append(d.strip(','))
                a, b, c, d = line[4].split(' ')
                Dane[10].append(a.strip(','))
                Dane[11].append(b.strip(','))
                Dane[12].append(c.strip(','))
                Dane[13].append(d.strip(','))
                a, b, c, d = line[5].split(' ')
                Dane[14].append(a.strip(','))
                Dane[15].append(b.strip(','))
                Dane[16].append(c.strip(','))
                Dane[17].append(d.strip(','))
                a, b, c, d = line[6].split(' ')
                Dane[18].append(a.strip(','))
                Dane[19].append(b.strip(','))
                Dane[20].append(c.strip(','))
                Dane[21].append(d.strip(','))
                a, b, c, d = line[7].split(' ')
                Dane[22].append(a.strip(','))
                Dane[23].append(b.strip(','))
                Dane[24].append(c.strip(','))
                Dane[25].append(d.strip(','))
                a, b, c, d = line[8].split(' ')
                Dane[26].append(a.strip(','))
                Dane[27].append(b.strip(','))
                Dane[28].append(c.strip(','))
                Dane[29].append(d.strip(','))


        Dane = np.array(Dane)

        a = Dane[1] == 'train'


        train = Dane[:, a]
        valid = Dane[:, ~a]

        train_bool = trainset

        if train_bool:
            dane = train
        else:
            dane = valid
        parameters=[dropout_val,momentum_val,lr,conv_dropout, stage]


        return dane,parameters
    else:
        return False,False

def calculate_mean_AUC(run, namespace='custom', epoch=200, trainset=False):
    dane, parameters = read_data(run, namespace=namespace, trainset=trainset)

    if type(dane) != type(False):
        dane=dane[14:]
        temp_dane=[]
        dane=np.array(dane)
        dane=dane[:,epoch:]

        for i in range(len(dane)):
            temp_dane.append(sum(list(map(float,dane[i,:])))/len(list(map(float,dane[i,:]))))

        dane=temp_dane

    return dane,parameters










def pamfl_mean_and_sd_of_many_runs_AUC(run_start,run_end,epoch=150,namespace='custom', train=False,
                                   cdrop=False, momentum_bool=False):


    results={}
    for run in range(run_start,run_end+1):

        dane,params = calculate_mean_AUC(run, namespace, epoch, train)
        if type(dane) != type(False):
            if cdrop:
                key_parameter = params[3]
                name_of_parameter = 'conv-dropout'
            elif momentum_bool:
                key_parameter = params[1]
                name_of_parameter = 'momentum'
            else:
                key_parameter = params[0]
                name_of_parameter = 'dropout'
            momentum = params[1]
            lr = params[2]
            dropout = params[0]
            conv_dropout = params[3]


            #AUC promoter active

            pa1=float(dane[0])
            pa2=float(dane[1])
            pa3=float(dane[2])
            pa4=float(dane[3])

            # AUC nonpromoter active

            npa1 = float(dane[4])
            npa2 = float(dane[5])
            npa3 = float(dane[6])
            npa4 = float(dane[7])

            # AUC promoter inactive

            pin1 = float(dane[8])
            pin2 = float(dane[9])
            pin3 = float(dane[10])
            pin4 = float(dane[11])

            # AUC nonpromoter inactive

            npin1 = float(dane[12])
            npin2 = float(dane[13])
            npin3 = float(dane[14])
            npin4 = float(dane[15])


            if key_parameter not in results:
                results[key_parameter]={'params':[dropout,momentum,conv_dropout,lr],'AUC_pa1':[pa1],
                                        'AUC_pa2': [pa2],'AUC_pa3':[pa3],'AUC_pa4':[pa4],
                            'AUC_npa1': [npa1],'AUC_npa2': [npa2], 'AUC_npa3': [npa3], 'AUC_npa4': [npa4],
                            'AUC_pin1': [pin1], 'AUC_pin2': [pin2], 'AUC_pin3': [pin3], 'AUC_pin4': [pin4],
                            'AUC_npin1': [npin1], 'AUC_npin2': [npin2], 'AUC_npin3': [npin3], 'AUC_npin4': [npin4]
                                 }
            else:
                results[key_parameter]['AUC_pa1'].append(pa1)
                results[key_parameter]['AUC_pa2'].append(pa2)
                results[key_parameter]['AUC_pa3'].append(pa3)
                results[key_parameter]['AUC_pa4'].append(pa4)

                results[key_parameter]['AUC_npa1'].append(npa1)
                results[key_parameter]['AUC_npa2'].append(npa2)
                results[key_parameter]['AUC_npa3'].append(npa3)
                results[key_parameter]['AUC_npa4'].append(npa4)

                results[key_parameter]['AUC_pin1'].append(pin1)
                results[key_parameter]['AUC_pin2'].append(pin2)
                results[key_parameter]['AUC_pin3'].append(pin3)
                results[key_parameter]['AUC_pin4'].append(pin4)

                results[key_parameter]['AUC_npin1'].append(npin1)
                results[key_parameter]['AUC_npin2'].append(npin2)
                results[key_parameter]['AUC_npin3'].append(npin3)
                results[key_parameter]['AUC_npin4'].append(npin4)









    for key in results.keys():
        results[key]['AUC_pa1']=sum(results[key]['AUC_pa1'])/len(results[key]['AUC_pa1'])
        results[key]['AUC_pa2']=sum(results[key]['AUC_pa2'])/len(results[key]['AUC_pa2'])
        results[key]['AUC_pa3']=sum(results[key]['AUC_pa3'])/len(results[key]['AUC_pa3'])
        results[key]['AUC_pa4']=sum(results[key]['AUC_pa4'])/len(results[key]['AUC_pa4'])

        results[key]['AUC_npa1']=sum(results[key]['AUC_npa1'])/len(results[key]['AUC_npa1'])
        results[key]['AUC_npa2']=sum(results[key]['AUC_npa2'])/len(results[key]['AUC_npa2'])
        results[key]['AUC_npa3']=sum(results[key]['AUC_npa3'])/len(results[key]['AUC_npa3'])
        results[key]['AUC_npa4']=sum(results[key]['AUC_npa4'])/len(results[key]['AUC_npa4'])

        results[key]['AUC_pin1']=sum(results[key]['AUC_pin1'])/len(results[key]['AUC_pin1'])
        results[key]['AUC_pin2']=sum(results[key]['AUC_pin2'])/len(results[key]['AUC_pin2'])
        results[key]['AUC_pin3']=sum(results[key]['AUC_pin3'])/len(results[key]['AUC_pin3'])
        results[key]['AUC_pin4']=sum(results[key]['AUC_pin4'])/len(results[key]['AUC_pin4'])

        results[key]['AUC_npin1']=sum(results[key]['AUC_npin1'])/len(results[key]['AUC_npin1'])
        results[key]['AUC_npin2']=sum(results[key]['AUC_npin2'])/len(results[key]['AUC_npin2'])
        results[key]['AUC_npin3']=sum(results[key]['AUC_npin3'])/len(results[key]['AUC_npin3'])
        results[key]['AUC_npin4']=sum(results[key]['AUC_npin4'])/len(results[key]['AUC_npin4'])

    x_axis=[]

    pa1=[]
    pa2=[]
    pa3=[]
    pa4=[]

    npa1 = []
    npa2 = []
    npa3 = []
    npa4 = []

    pin1 = []
    pin2 = []
    pin3 = []
    pin4 = []

    npin1 = []
    npin2 = []
    npin3 = []
    npin4 = []

    for key in results.keys():
        x_axis.append(float(key))

        params=results[key]['params']

        pa1.append(results[key]['AUC_pa1'])
        pa2.append(results[key]['AUC_pa2'])
        pa3.append(results[key]['AUC_pa3'])
        pa4.append(results[key]['AUC_pa4'])

        npa1.append(results[key]['AUC_npa1'])
        npa2.append(results[key]['AUC_npa2'])
        npa3.append(results[key]['AUC_npa3'])
        npa4.append(results[key]['AUC_npa4'])

        pin1.append(results[key]['AUC_pin1'])
        pin2.append(results[key]['AUC_pin2'])
        pin3.append(results[key]['AUC_pin3'])
        pin4.append(results[key]['AUC_pin4'])

        npin1.append(results[key]['AUC_npin1'])
        npin2.append(results[key]['AUC_npin2'])
        npin3.append(results[key]['AUC_npin3'])
        npin4.append(results[key]['AUC_npin4'])

    sort_index = sorted(range(len(x_axis)), key=lambda k: x_axis[k])
    sort_index = [x for _, x in sorted(zip(sort_index, range(len(sort_index))))]

    x_axis.sort()

    pa1 = [x for _, x in sorted(zip(sort_index, pa1))]
    pa2 = [x for _, x in sorted(zip(sort_index, pa2))]
    pa3 = [x for _, x in sorted(zip(sort_index, pa3))]
    pa4 = [x for _, x in sorted(zip(sort_index, pa4))]

    npa1 = [x for _, x in sorted(zip(sort_index, npa1))]
    npa2 = [x for _, x in sorted(zip(sort_index, npa2))]
    npa3 = [x for _, x in sorted(zip(sort_index, npa3))]
    npa4 = [x for _, x in sorted(zip(sort_index, npa4))]

    pin1 = [x for _, x in sorted(zip(sort_index, pin1))]
    pin2 = [x for _, x in sorted(zip(sort_index, pin2))]
    pin3 = [x for _, x in sorted(zip(sort_index, pin3))]
    pin4 = [x for _, x in sorted(zip(sort_index, pin4))]

    npin1 = [x for _, x in sorted(zip(sort_index, npin1))]
    npin2 = [x for _, x in sorted(zip(sort_index, npin2))]
    npin3 = [x for _, x in sorted(zip(sort_index, npin3))]
    npin4 = [x for _, x in sorted(zip(sort_index, npin4))]

    results=[pa1,pa2,pa3,pa4,npa1,npa2,npa3,npa4,pin1,pin2,pin3,pin4,npin1,npin2,npin3,npin4]


    return results, name_of_parameter, params




results, name_of_parameter, params = pamfl_mean_and_sd_of_many_runs_AUC(36,59,epoch=100,namespace='custom', train=False,
                                   cdrop=False, momentum_bool=False)

