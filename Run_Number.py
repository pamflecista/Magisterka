import os
from pathlib import Path
import re
import os.path

networkRegex=re.compile(r'(custom|basset)([0-9]{1,5})')
path_res=Path.cwd()/'results'
path_txt=Path.cwd()/'number.txt'
path_paths=Path.cwd()/'paths.txt'

path_to_script=Path.cwd()/'magisterka'/'train.py'
path_to_main=Path.cwd()
command=' '.join(['python ',str(path_to_script),'dataset3','--path',str(path_to_main),'--num_epochs 1',
                 '--network custom --run '])
with open(path_paths, "w") as f:
    f.write(str(command).rstrip())



if os.path.isdir(path_res):
    numbers=[int(networkRegex.search(name).group(2)) for name in
             os.listdir(path_res)
             if networkRegex.search(name) is not None]
    number=max(numbers)+1
    with open(path_txt,"w") as f:
        f.write(str(number))
else:
    with open(path_txt,"w") as f:
        f.write(str(1))




