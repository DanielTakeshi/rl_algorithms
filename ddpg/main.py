"""
Main script for DDPG code, for CONTINUOUS control environments.

(c) 2017 by Daniel Seita, though mostly building upon other code as usual, with
credit attrbuted to in the DDPG's README.
"""

from ddpg import DDPGAgent
import argparse
import gym
import numpy as np
np.set_printoptions(suppress=True, precision=5, edgeitems=10)
import pickle
import sys
import tensorflow as tf
if "../" not in sys.path:
    sys.path.append("../")
from utils import utils_pg as utils
from utils import value_functions as vfuncs
from utils import logz
from utils import policies


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('envname', type=str)

    # DDPG stuff, all directly from the paper.
    p.add_argument('--batch_size', type=int, default=64) # Used 16 on pixels.
    p.add_argument('--ou_noise_theta', type=float, default=0.15)
    p.add_argument('--ou_noise_sigma', type=float, default=0.2)
    p.add_argument('--Q_gamma', type=float, default=0.99)
    p.add_argument('--Q_l2_weight_decay', type=float, default=1e-2)
    p.add_argument('--replay_size', type=int, default=1000000)
    p.add_argument('--step_size_actor', type=float, default=1e-4)
    p.add_argument('--step_size_critic', type=float, default=1e-3)
    p.add_argument('--tau', type=float, default=0.001)

    # Other stuff that I use for my own or based on other code.
    p.add_argument('--do_not_save', action='store_true')
    p.add_argument('--n_iter', type=int, default=10000)
    p.add_argument('--learning_freq', type=int, default=50)
    p.add_argument('--log_every_t_iter', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--wait_until_rbuffer', type=int, default=1000)
    args = p.parse_args()

    # Handle the log directory and save the arguments.
    logdir = 'out/' +args.envname+ '/seed' +str(args.seed).zfill(2)
    if args.do_not_save:
        logdir = None
    logz.configure_output_dir(logdir)
    if logdir is not None:
        with open(logdir+'/args.pkl', 'wb') as f:
            pickle.dump(args, f)
    print("Saving in logdir: {}".format(logdir))

    # Other stuff for seeding and getting things set up.
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(args.envname)
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, 
                               intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)

    ddpg = DDPGAgent(sess, env, args)
    ddpg.train()
    ddpg.test()
