import bin.common
import os
from pathlib import Path
number=bin.common.run_number()
path_to_script=Path.cwd()/'train.py'
path_to_main=Path.cwd().parents[0]
command=' '.join(['python ',str(path_to_script),'dataset3','--path',str(path_to_main),'--num_epochs 200',
                     '--network custom --run {}'])

for i in range(3):
    os.system(command.format(number+i)
)
