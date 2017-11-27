import logging
import pickle

import h5py
import numpy as np
import tensorflow as tf

try:
    from . import tf_util as U
except:
    import tf_util  as U


import json
import numpy as np



GRID_SIZE = 10

logger = logging.getLogger(__name__)

import numpy as np
import logging
import pickle

import h5py
import numpy as np
import tensorflow as tf
from random import sample as rsample

# from . import tf_util as U



logger = logging.getLogger(__name__)


class Policy:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.scope = self._initialize(*args, **kwargs)
        self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, self.scope.name)

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
        self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        self._setfromflat = U.SetFromFlat(self.trainable_variables)
        self._getflat = U.GetFlat(self.trainable_variables)

        # logger.info('Trainable variables ({} parameters)'.format(self.num_params))
        for v in self.trainable_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))


        logger.info('All variables')
        for v in self.all_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))

        placeholders = [tf.placeholder(v.value().dtype, v.get_shape().as_list()) for v in self.all_variables]
        self.set_all_vars = U.function(
            inputs=placeholders,
            outputs=[],
            updates=[tf.group(*[v.assign(p) for v, p in zip(self.all_variables, placeholders)])]
        )

    def _initialize(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, filename):
        assert filename.endswith('.h5')
        with h5py.File(filename, 'w') as f:
            for v in self.all_variables:
                f[v.name] = v.eval()
            # TODO: it would be nice to avoid pickle, but it's convenient to pass Python objects to _initialize
            # (like Gym spaces or numpy arrays)
            f.attrs['name'] = type(self).__name__
            f.attrs['args_and_kwargs'] = np.void(pickle.dumps((self.args, self.kwargs), protocol=-1))

    @classmethod
    def Load(cls, filename, extra_kwargs=None):
        with h5py.File(filename, 'r') as f:
            args, kwargs = pickle.loads(f.attrs['args_and_kwargs'].tostring())
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            policy = cls(*args, **kwargs)
            policy.set_all_vars(*[f[v.name][...] for v in policy.all_variables])
        return policy

    def act(self, ob, random_stream=None):
        raise NotImplementedError

    def set_trainable_flat(self, x):
        self._setfromflat(x)

    def get_trainable_flat(self):
        return self._getflat()

    @property
    def needs_ob_stat(self):
        raise NotImplementedError

    def set_ob_stat(self, ob_mean, ob_std):
        raise NotImplementedError


class CatchPolicy_off_poliy(Policy):
    def _initialize(self, ob_space, ac_space, nonlin_type, hidden_dims, connection_type):
        self.ac_space = ac_space
        self.hidden_dims = hidden_dims
        self.connection_type = connection_type

        assert len(ob_space.shape) == len(self.ac_space.shape) == 1

        self.nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'elu': tf.nn.elu}[nonlin_type]

        with tf.variable_scope(type(self).__name__) as scope:
            # Policy network
            o = tf.placeholder(tf.float32, [None] + list(ob_space.shape))
            a = self._make_net(o)
            self._act = U.function([o], a)
        return scope

    def _make_net(self, o):
        # Process observation
        if self.connection_type == 'ff':
            x = o
            for ilayer, hd in enumerate(self.hidden_dims):
                x = self.nonlin(U.dense(x, hd, 'l{}'.format(ilayer), U.normc_initializer(1.0)))
        else:
            raise NotImplementedError(self.connection_type)

        # Map to action
        scores = U.dense(x, 3, 'out', U.normc_initializer(0.01))
        # scores_nab = tf.reshape(scores, [-1, 1, 3])
        # aidx_na =  tf.argmax(scores_nab, 2)  # 0 ... num_bins-1
        #a = tf.to_float(scores)
        #sft = tf.nn.softmax(scores)

        return tf.to_float(scores)

    def initialize_from(self, filename, ob_stat=None):
        """
        Initializes weights from another policy, which must have the same architecture (variable names),
        but the weight arrays can be smaller than the current policy.
        """
        with h5py.File(filename, 'r') as f:
            f_var_names = []
            f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'

            init_vals = []
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                f_shp = f[v.name].shape
                assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
                    'This policy must have more weights than the policy to load'
                init_val = v.eval()
                # ob_mean and ob_std are initialized with nan, so set them manually
                if 'ob_mean' in v.name:
                    init_val[:] = 0
                    init_mean = init_val
                elif 'ob_std' in v.name:
                    init_val[:] = 0.001
                    init_std = init_val
                # Fill in subarray from the loaded policy
                init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
                init_vals.append(init_val)
            self.set_all_vars(*init_vals)

        if ob_stat is not None:
            ob_stat.set_from_init(init_mean, init_std, init_count=1e5)



       # === Rollouts/training ===

    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        env_timestep_limit = GRID_SIZE - 2
        timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)


        trajWeightedReards = []
        for i in range(40):
            traj = env.get_trajectory()
            percOfSame = 0.

            weightedReward = 0
            for step_no in range(timestep_limit):
                ob, teacherAction, rew, teacherAcDistr = traj[step_no]
                ac = self.act(ob[None], random_stream=random_stream)[0]

                polAcDistr = np.array(ac)

                if np.argmax(polAcDistr) == teacherAction:
                    percOfSame+=1.

                weightedReward += (percOfSame/ (step_no + 1))* rew

            trajWeightedReards.append(weightedReward)

        return np.array([np.mean(trajWeightedReards)]), timestep_limit



    def act(self, ob, random_stream=None):
        return self._act(ob)

    @property
    def needs_ob_stat(self):
        return False

    @property
    def needs_ref_batch(self):
        return False



class catcher():
    def __init__(self):
        #self.memory = pickle.load(open('/home/alexey/experiments/evolution-strategies-starter-master/es_distributed/memoryList.pickle', 'rb'))

        self.observation_space = np.zeros((GRID_SIZE, GRID_SIZE)).ravel()
        self.action_space = np.array([2])

    def step(self, ac):
        return self.ep.send(ac)


    def reset(self):
        self.ep = episode()
        S, won, _, _ = self.ep.__next__()
        return S


class CatchPolicy(Policy):
    def _initialize(self, ob_space, ac_space, nonlin_type, hidden_dims, connection_type):
        self.ac_space = ac_space
        self.hidden_dims = hidden_dims
        self.connection_type = connection_type

        assert len(ob_space.shape) == len(self.ac_space.shape) == 1

        self.nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'elu': tf.nn.elu}[nonlin_type]

        with tf.variable_scope(type(self).__name__) as scope:
            # Policy network
            o = tf.placeholder(tf.float32, [None] + list(ob_space.shape))
            a = self._make_net(o)
            self._act = U.function([o], a)
        return scope

    def _make_net(self, o):
        # Process observation
        if self.connection_type == 'ff':
            x = o
            for ilayer, hd in enumerate(self.hidden_dims):
                x = self.nonlin(U.dense(x, hd, 'l{}'.format(ilayer), U.normc_initializer(1.0)))
        else:
            raise NotImplementedError(self.connection_type)

        # Map to action
        scores = U.dense(x, 3, 'out', U.normc_initializer(0.01))
        scores_nab = tf.reshape(scores, [-1, 1, 3])
        aidx_na =  tf.argmax(scores_nab, 2)  # 0 ... num_bins-1
        #a = tf.to_float(scores)
        #sft = tf.nn.softmax(scores)

        return aidx_na

    def initialize_from(self, filename, ob_stat=None):
        """
        Initializes weights from another policy, which must have the same architecture (variable names),
        but the weight arrays can be smaller than the current policy.
        """
        with h5py.File(filename, 'r') as f:
            f_var_names = []
            f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'

            init_vals = []
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                f_shp = f[v.name].shape
                assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
                    'This policy must have more weights than the policy to load'
                init_val = v.eval()
                # ob_mean and ob_std are initialized with nan, so set them manually
                if 'ob_mean' in v.name:
                    init_val[:] = 0
                    init_mean = init_val
                elif 'ob_std' in v.name:
                    init_val[:] = 0.001
                    init_std = init_val
                # Fill in subarray from the loaded policy
                init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
                init_vals.append(init_val)
            self.set_all_vars(*init_vals)

        if ob_stat is not None:
            ob_stat.set_from_init(init_mean, init_std, init_count=1e5)



       # === Rollouts/training ===

    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        #env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        #timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
        timestep_limit = GRID_SIZE
        rews = []
        t = 0
        if save_obs:
            obs = []
        ob = env.reset()
        for _ in range(timestep_limit):
            ac = int(self.act(ob[None], random_stream=random_stream)[0])
            if save_obs:
                obs.append(ob)
            ob, rew, done, _ = env.step(ac)
            rews.append(rew)
            t += 1
            if render:
                env.render()
            if done:
                break
        rews = np.array(rews, dtype=np.float32)
        if save_obs:
            return rews, t, np.array(obs)

        return rews, t



    def act(self, ob, random_stream=None):
        return self._act(ob)

    @property
    def needs_ob_stat(self):
        return False

    @property
    def needs_ref_batch(self):
        return False


def episode():
    """
    Coroutine of episode.

    Action has to be explicitly send to this coroutine.
    """
    x, y, z = (
        np.random.randint(0, GRID_SIZE),  # X of fruit
        0,  # Y of dot
        np.random.randint(1, GRID_SIZE - 1)  # X of basket
    )
    while True:
        X = np.zeros((GRID_SIZE, GRID_SIZE))  # Reset grid
        X[y, x] = 1.  # Draw fruit
        bar = range(z - 1, z + 2)
        X[-1, bar] = 1.  # Draw basket

        # End of game is known when fruit is at penultimate line of grid.
        # End represents either a win or a loss
        end = int(y >= GRID_SIZE - 2)
        rew = 0
        if end and x in bar:
            rew = 1000
        if end and x not in bar:
            rew = -1000
        if end and x == bar[1]:
            rew = 1500

        move = yield X.ravel(), rew, end, None
        if end:
            break
        z = np.min([np.max([z + move - 1, 1]), GRID_SIZE - 2])
        y += 1




