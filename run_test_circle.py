import socket
import numpy as np
import time
from pathlib import Path
import subprocess
import os
import errno

bashCommand = "squeue  -p test | grep ignatov"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)

output, error = process.communicate()
print(str(output))

#process = subprocess.Popen(bashCommand.split(), stdout=stdout, shell=True    )