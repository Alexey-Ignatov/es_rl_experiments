#!/bin/sh
NAME=exp_`date "+%m_%d_%H_%M_%S"`
EXP_FILE=$1
tmux new -s $NAME -d
tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'source ./../es_rl_env/bin/activate' C-m
tmux send-keys -t $NAME 'python -m es_distributed.main master --exp_file ./configurations/humanoid.json --master_socket_path /tmp/es_redis_master.sock'"$EXP_FILE" C-m
tmux split-window -t $NAME

tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'source ./../es_rl_env/bin/activate' C-m
tmux send-keys -t $NAME 'python -m es_distributed.main workers --master_host localhost --relay_socket_path /tmp/es_redis_relay.sock --num_workers 8' C-m
tmux a -t $NAME
/Users/xxx/rl_es/evolution-strategies-starter-master/configurations/humanoid.json