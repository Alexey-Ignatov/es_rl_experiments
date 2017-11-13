import socket
import numpy as np
import time
from pathlib import Path
import subprocess
import os
import errno

start_dir = "/mnt/data/users/dm4/vol12/ignatovalexey_1956/_scratch/es_rl_experiments/evolution-strategies-starter-master"
log_dir = "/mnt/data/users/dm4/vol12/ignatovalexey_1956/_scratch/logs"
redis_server_path = "/mnt/data/users/dm4/vol12/ignatovalexey_1956/_scratch/redis-stable/src/redis-server"
master_ip_path = '/mnt/data/users/dm4/vol12/ignatovalexey_1956/_scratch/master_ip.txt'
tmp_dir = "/mnt/data/users/dm4/vol12/ignatovalexey_1956/_scratch/dir"


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def run_coms_list(bashCommands, cwd):
    for bashCommand, wait in bashCommands:
        stdout = subprocess.PIPE if wait else None
        process = subprocess.Popen(bashCommand.split(), stdout=stdout, cwd = cwd)
        if wait:
            output, error = process.communicate()
            print(str(output))

def get_worker_coms_list(master_ip):
    return [
            ("{} {}/redis_config/redis_local_mirror.conf".format(redis_server_path, start_dir), False),
            ("python -m es_distributed.main workers --master_host {} --relay_socket_path /tmp/es_redis_relay.sock --num_workers 1".format( master_ip), True)]


def get_master_coms_list():
    return [
            ("{} {}/redis_config/redis_master.conf".format(redis_server_path, start_dir), False),
            ("python -m es_distributed.main master --exp_file ./configurations/humanoid.json --master_socket_path /tmp/es_redis_master.sock --log_dir {}".format(log_dir), True)]



def get_my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    master_ip = s.getsockname()[0]
    s.close()
    return master_ip



my_ip = get_my_ip()
time.sleep(np.random.randint(1000)*.01)

master_ip_file = Path(master_ip_path)
if not master_ip_file.is_file():
    with open(master_ip_path, 'w') as f:
        f.write(my_ip)
        
    run_coms_list(get_master_coms_list(), start_dir)

else:
    with open(master_ip_path) as f:
        master_ip = f.read()

    mkdir_p(tmp_dir)
    
    my_ip_file = Path(tmp_dir + '/' + my_ip)
    if not my_ip_file.is_file():
        with open(tmp_dir + '/' + my_ip, 'w') as f:
            f.write(my_ip)

        relay_run_com = "python -m es_distributed.main relay --master_host localhost --relay_socket_path /tmp/es_redis_relay.sock"
        subprocess.Popen(relay_run_com.split(),stdout=subprocess.PIPE, cwd = start_dir)

    run_coms_list(get_worker_coms_list(master_ip), start_dir)

        

