import socket
import numpy as np
import time
from pathlib import Path
import subprocess

start_dir = "/mnt/data/users/dm4/vol12/ignatovalexey_1956/_scratch/es_rl_experiments/evolution-strategies-starter-master"
log_dir = "/mnt/data/users/dm4/vol12/ignatovalexey_1956/_scratch/logs"
redis_server_path = "/mnt/data/users/dm4/vol12/ignatovalexey_1956/_scratch/redis-stable/src/redis-server"

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
            ("python -m es_distributed.main workers --master_host {} --relay_socket_path /tmp/es_redis_relay.sock --num_workers 8".format( master_ip), False)]


def get_master_coms_list(master_ip):
    return [
            ("{} {}/redis_config/redis_master.conf".format(redis_server_path, start_dir), False),
            ("python -m es_distributed.main master --exp_file ./configurations/humanoid.json --master_socket_path /tmp/es_redis_master.sock --log_dir {}".format(log_dir), False)]


master_id_path = '/mnt/data/users/dm4/vol12/ignatovalexey_1956/_scratch/master_ip.txt'

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
master_ip = s.getsockname()[0]
s.close()

time.sleep(np.random.randint(1000)*.001)

my_file = Path(master_id_path)
if not my_file.is_file():
    with open(master_id_path, 'w') as f:
        f.write(master_ip)
        
    run_coms_list(get_master_coms_list(master_ip), start_dir) 

else:
    with open(master_id_path) as f:
        master_ip = f.read()
    print(type(master_ip), master_ip, 'master_ip')
    
    
    
    run_coms_list(get_worker_coms_list(master_ip), start_dir)    
        
        

