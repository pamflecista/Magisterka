import bin.common
import os
from pathlib import Path
command1= 'mkdir MgrPtit/sRes/pamfl_v2_{x}'
command2='cp MgrPtit/results/pamfl_v2_{x}/pamfl_v2_{x}_pamfl_params.csv MgrPtit/sRes/pamfl_v2_{x}'
command3='cp MgrPtit/results/pamfl_v2_{x}/pamfl_v2_{x}_train_results.tsv MgrPtit/sRes/pamfl_v2_{x}'

for i in range(1, 8):
    os.system(command1.format(x=i))
    os.system(command2.format(x=i))
    os.system(command3.format(x=i))

command1= 'cd MgrPtit/sRes/'
command2= 'git add pamfl_v2_{x}'

for i in range(1, 8):
    os.system(command1.format(x=i))
    os.system(command2.format(x=i))
    

