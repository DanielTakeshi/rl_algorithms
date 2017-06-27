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

    # Actor, Critic, and Replay Buffer stuff
    p.add_argument('--actor_step_size', type=float, default=1e-4)
    p.add_argument('--critic_step_size', type=float, default=1e-3)
    p.add_argument('--replay_size', type=int, default=1000000)

    # Other stuff
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--do_not_save', action='store_true')
    p.add_argument('--n_iter', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    # Handle the log directory and save the arguments.
    logdir = 'outputs/' +args.envname+ '/seed' +str(args.seed).zfill(2)
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
