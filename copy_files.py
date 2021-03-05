import bin.common
import os
from pathlib import Path
command1= 'mkdir MgrPtit/sRes/pamfl{x}'
command2='cp MgrPtit/results/pamfl{x}/pamfl{x}_pamfl_params.csv MgrPtit/sRes/pamfl{x}'
command3='cp MgrPtit/results/pamfl{x}/pamfl{x}_train_results.tsv MgrPtit/sRes/pamfl{x}'

for i in range(19, 20):
    os.system(command1.format(x=i))
    os.system(command2.format(x=i))
    os.system(command3.format(x=i))
