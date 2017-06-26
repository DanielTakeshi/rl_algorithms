"""
Vanilla Policy Gradients, aka REINFORCE, aka Monte Carlo Policy Gradients.

To quickly test you can do:

    python main.py Pendulum-v0 --vf_type nn --use_kl_heuristic --do_not_save

As long as --do_not_save is there, it won't overwrite files.  If I want to
benchmark and save results, see the bash scripts. Add --render if desired.

(c) April 2017 by Daniel Seita, built upon starter code from CS 294-112.
"""

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


def run_vpg(args, vf_params, logdir, env, sess, continuous_control):
    """ General purpose method to run vanilla policy gradients, for both
    continuous and discrete action environments. 
    
    Parameters
    ----------
    args: [Namespace]
        Contains user-provided (or default) arguments for VPGs.
    vf_params: [dict]
        Dictionary of parameters for the value function.
    logdir: [string]
        Where we store the outputs, can be None to avoid saving.
    env: [OpenAI gym env]
        The environment the agent is in, from OpenAI gym.
    sess: [tf Session]
        Current Tensorflow session, to be passed to (at least) the policy
        function, and the value function as well if it's a neural network.
    continuous_control: [boolean]
        True if continuous control (i.e. actions), false if otherwise.
    """
    ob_dim = env.observation_space.shape[0]

    if args.vf_type == 'linear':
        vf = vfuncs.LinearValueFunction(**vf_params)
    elif args.vf_type == 'nn':
        vf = vfuncs.NnValueFunction(session=sess, ob_dim=ob_dim, **vf_params)

    if continuous_control:
        ac_dim = env.action_space.shape[0]
        policyfn = policies.GaussianPolicy(sess, ob_dim, ac_dim)
    else:
        ac_dim = env.action_space.n
        policyfn = policies.GibbsPolicy(sess, ob_dim, ac_dim)

    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101
    total_timesteps = 0
    stepsize = args.initial_stepsize

    for i in range(args.n_iter):
        print("\n********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps.
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode = (len(paths) == 0 and (i%100 == 0) and args.render)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = policyfn.sample_action(ob)
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break                    
            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += utils.pathlength(path)
            if timesteps_this_batch > args.min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Estimate advantage function using baseline vf (these are lists!).
        # return_t: list of sum of discounted rewards (to end of episode), one per time
        # vpred_t: list of value function's predictions of components of return_t
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = utils.discount(rew_t, args.gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update and **re-fit the baseline**.
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n  = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        std_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        # Policy update, plus diagnostics stuff. Is there a better way to handle
        # the continuous vs discrete control cases?
        if continuous_control:
            surr_loss, oldmean_na, oldlogstd_a = policyfn.update_policy(
                    ob_no, ac_n, std_adv_n, stepsize)
            kl, ent = policyfn.kldiv_and_entropy(ob_no, oldmean_na, oldlogstd_a)
        else:
            surr_loss, oldlogits_na = policyfn.update_policy(
                    ob_no, ac_n, std_adv_n, stepsize)
            kl, ent = policyfn.kldiv_and_entropy(ob_no, oldlogits_na)

        # A step size heuristic to ensure that we don't take too large steps.
        if args.use_kl_heuristic:
            if kl > args.desired_kl * 2: 
                stepsize /= 1.5
                print('PG stepsize -> %s' % stepsize)
            elif kl < args.desired_kl / 2: 
                stepsize *= 1.5
                print('PG stepsize -> %s' % stepsize)
            else:
                print('PG stepsize OK')

        # Log diagnostics
        if i % args.log_every_t_iter == 0:
            logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
            logz.log_tabular("EpLenMean", np.mean([utils.pathlength(path) for path in paths]))
            logz.log_tabular("KLOldNew", kl)
            logz.log_tabular("Entropy", ent)
            logz.log_tabular("EVBefore", utils.explained_variance_1d(vpred_n, vtarg_n))
            logz.log_tabular("EVAfter", utils.explained_variance_1d(vf.predict(ob_no), vtarg_n))
            logz.log_tabular("SurrogateLoss", surr_loss)
            logz.log_tabular("TimestepsSoFar", total_timesteps)
            # If you're overfitting, EVAfter will be way larger than EVBefore.
            # Note that we fit the value function AFTER using it to compute the
            # advantage function to avoid introducing bias
            logz.dump_tabular()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('envname', type=str)
    p.add_argument('--render', action='store_true')
    p.add_argument('--do_not_save', action='store_true')
    p.add_argument('--use_kl_heuristic', action='store_true')

    p.add_argument('--n_iter', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--gamma', type=float, default=0.97)
    p.add_argument('--desired_kl', type=float, default=2e-3)
    p.add_argument('--min_timesteps_per_batch', type=int, default=2500) 
    p.add_argument('--initial_stepsize', type=float, default=1e-3)
    p.add_argument('--log_every_t_iter', type=int, default=1)

    p.add_argument('--vf_type', type=str, default='linear')
    p.add_argument('--nnvf_epochs', type=int, default=20)
    p.add_argument('--nnvf_ssize', type=float, default=1e-3)
    args = p.parse_args()

    # Handle value function type and the log directory (and save the args!).
    assert args.vf_type == 'linear' or args.vf_type == 'nn'
    vf_params = {}
    outstr = 'linearvf-kl' +str(args.desired_kl) 
    if args.vf_type == 'nn':
        vf_params = dict(n_epochs=args.nnvf_epochs, stepsize=args.nnvf_ssize)
        outstr = 'nnvf-kl' +str(args.desired_kl)
    outstr += '-seed' +str(args.seed).zfill(2)
    logdir = 'outputs/' +args.envname+ '/' +outstr
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
    continuous = True
    if 'discrete' in str(type(env.action_space)).lower():
        # A bit of a hack, is there a better way to do this?  Another option
        # could be following Jonathan Ho's code and detecting spaces.Box?
        continuous = False
    print("Continuous control? {}".format(continuous))
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, 
                               intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)

    run_vpg(args, vf_params, logdir, env, sess, continuous)
