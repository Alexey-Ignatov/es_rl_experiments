import socket
import numpy as np
import time
from pathlib import Path
import subprocess
import os
import errno
import glob
import os


master_ip_path = '/mnt/data/users/dm4/vol12/ignatovalexey_1956/_scratch/master_ip.txt'
sqCommand = "squeue  -p test | grep ignatov"
run_on_test = 'sbatch -p test -n128 run python /mnt/data/users/dm4/vol12/ignatovalexey_1956/_scratch/es_rl_experiments/run_es.py'

#sqCommand = "ls | grep e"
try:
    procs = len(subprocess.check_output(sqCommand, shell=True   ).splitlines())
except subprocess.CalledProcessError:
    procs = 0


if procs == 0:
    try:
        subprocess.check_output('rm /mnt/data/users/dm4/vol12/ignatovalexey_1956/_scratch/tmp/*', shell=True)
    except subprocess.CalledProcessError:
        pass

    try:
        subprocess.check_output('rm '+ master_ip_path, shell=True)
    except subprocess.CalledProcessError:
        pass

    print(subprocess.check_output(run_on_test, shell=True))
else:
    print('some tasks already in test')
#output, error = process.communicate()
#print(str(output))

#process = subprocess.Popen(bashCommand.split(), stdout=stdout, shell=True    )
#process = subprocess.Popen(bashCommand.split(), stdout=stdout, shell=True    )