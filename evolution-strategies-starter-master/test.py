import es_distributed.tf_util as U
import tensorflow as tf

from es_distributed.policies import catcher, CatchPolicy
from es_distributed.es import *
from es_distributed import policies


exp = {
  "config": {
    "calc_obstat_prob": 0.0,
    "episodes_per_batch": 10000,
    "eval_prob": 0.03,
    "l2coeff": 0.005,
    "noise_stdev": 0.02,
    "snapshot_freq": 5,
    "timesteps_per_batch": 100000,
    "return_proc_mode": "centered_rank",
    "episode_cutoff_mode": "env_default"
  },
  "env_id": "catcher",
  "exp_prefix": "humanoid",
  "optimizer": {
    "args": {
      "stepsize": 0.01
    },
    "type": "adam"
  },
  "policy": {
    "args": {
      "connection_type": "ff",
      "hidden_dims": [
        100,
        100
      ],
      "nonlin_type": "tanh"
    },
    "type": "CatchPolicy"
  }
}


config, env, sess, policy = setup(exp, single_threaded=False)