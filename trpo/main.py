"""
This is the main point for the Trust Region Policy Optimization (TRPO)
algorithm. Call this code using one of the bash scripts in my repository.
Otherwise, to quickly test without saving, use:

    python main.py Pendulum-v0 --vf_type nn --do_not_save --render

Disable --render if this is over SSH!! (There might be a way around this, I'm
not sure.)

(c) April 2017 by Daniel Seita. This code is built upon starter code from
Berkeley CS 294-112.
"""

import argparse
import gym
import itertools
import numpy as np
import sys
import tensorflow as tf
import time
if "../" not in sys.path:
    sys.path.append("../")
from utils import logz
from fxn_approx import *
from trpo import *


def run_trpo_algorithm(args, vf_params, logdir):
    """ Runs TRPO and prints/saves the result as needed.

    Params:
        args: Contains a LOT of user-provided arguments.
        vf_params: Parameters for the value function we're using.
        logdir: Where to save the output, if desired.
    """
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(args.envname)
    logz.configure_output_dir(logdir)

    # Create `sess` here so that we can pass it to the NN value function.
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, 
                               intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)

    # Create the TRPO agent, which will also construct its computational graph.
    TRPOAgent = TRPO(args, sess, env, vf_params)

    # Now some administration to get things started.
    sess.__enter__()
    tf.global_variables_initializer().run() #pylint: disable=E1101
    stepsize = args.initial_stepsize
    tstart = time.time()
    seed_iter = itertools.count()

    # Official TRPO iterations.
    for i in range(args.n_iter):
        print("********** iteration %i ************"%i)
        infodict = {}
        vfdict = {}
        paths = TRPOAgent.get_paths(seed_iter, env)
        TRPOAgent.compute_advantages(paths)
        TRPOAgent.fit_value_function(paths, vfdict)
        TRPOAgent.update_policy(paths, infodict)
        TRPOAgent.log_diagnostics(paths, infodict, vfdict)
    print("\nAll done!")


if __name__ == "__main__":
    # Get all the major arguments set up for TRPO here.
    p = argparse.ArgumentParser()
    p.add_argument('envname', type=str)
    p.add_argument('--cg_damping', type=float, default=0.1)
    p.add_argument('--do_not_save', action='store_true')
    p.add_argument('--gamma', type=float, default=0.98)
    p.add_argument('--initial_stepsize', type=float, default=1e-3)
    p.add_argument('--max_kl', type=float, default=0.01)
    p.add_argument('--min_timesteps_per_batch', type=int, default=5000) 
    p.add_argument('--n_iter', type=int, default=250)
    p.add_argument('--nnvf_epochs', type=int, default=50)
    p.add_argument('--nnvf_ssize', type=float, default=1e-3)
    p.add_argument('--render', action='store_true')
    p.add_argument('--render_frequency', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--vf_type', type=str, default='linear')
    args = p.parse_args()
    print("\nRunning TRPO with args:\n{}\n".format(args.__dict__))

    assert args.vf_type == 'linear' or args.vf_type == 'nn'
    vf_params = {}
    outstr = 'linearvf-kl' +str(args.max_kl) 
    if args.vf_type == 'nn':
        vf_params = dict(n_epochs=args.nnvf_epochs, stepsize=args.nnvf_ssize)
        outstr = 'nnvf-kl' +str(args.max_kl)
    outstr += '-cg' +str(args.cg_damping)
    outstr += '-seed' +str(args.seed).zfill(2)
    logdir = 'outputs/' +args.envname+ '/' +outstr
    if args.do_not_save:
        logdir = None

    run_trpo_algorithm(args, vf_params, logdir)
