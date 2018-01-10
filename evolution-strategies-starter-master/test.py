import es_distributed.tf_util as U
import tensorflow as tf
from es_distributed.policies import catcher, CatchPolicy


env = catcher()
cp = CatchPolicy(env.observation_space ,env.action_space, 'tanh', [100, 100], 'ff')
U.initialize()