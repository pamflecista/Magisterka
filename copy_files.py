import bin.common
import os
from pathlib import Path
command1= 'mkdir ../sRes/pamfl{x}'
command2='cp ../results/pamfl{x}/pamfl{x}_pamfl_params.csv ../sRes/pamfl{x}'
command3='cp ../results/pamfl{x}/pamfl{x}_train_results.tsv ../sRes/pamfl{x}'

for i in range(10, 19):
    os.system(command1.format(x=i))
    os.system(command2.format(x=i))
    os.system(command3.format(x=i))
