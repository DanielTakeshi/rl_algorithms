"""
Vanilla Policy Gradients, aka REINFORCE, aka Monte Carlo Policy Gradients.

To quickly test:

    python main.py Pendulum-v0 --vf_type nn --do_not_save

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


def vpg_continuous(logdir, args, vf_params):
    """ TODO this method will be deleted soon.
    
    Runs policy gradients on environments with continuous action spaces.

    Convention for symbolic construction naming/shapes: use `n` (batch size),
    `o`, and/or `a`.  The following are for both the gradient and the policy:

      sy_ob_no:      batch of observations, both for PG computation AND running policy
      sy_h1/sy_h2:   both are hidden layers with leaky-relus.
      sy_mean_na:    final net output (like sy_logits_na from earlier), mean of a Gaussian
      sy_n:          clever way to obtain the batch size (or 1, for running policy)

    For the policy, NOT the gradient:

      sy_sampled_ac: the current sampled action (a vector of controls) when running policy

    For the gradient, NOT the policy:

      sy_ac_na:      batch of actions taken by the policy, for PG computation
      sy_adv_n:      advantage function estimate (one per action vector)
      sy_logprob_n:  log-prob of actions taken in the batch, for PG computation

    Parameters are (neural network weights, log std vector). The policy network
    will output the _mean_ of a Gaussian, NOT our actual action.  The mean,
    along with the log std vector, defines a Gaussian which we sample from to
    get the action vector the agent plays. Thus, sy_mean_na is NOT a parameter.

    Args:
        logdir: The place where logs are sent to.
        args: Contains a LOT of parameters.
        vf_params: Paramters for the value function approximator.
    """
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make(args.envname)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    logz.configure_output_dir(logdir)

    # Create `sess` here so that we can pass it to the NN value function.
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)
    if args.vf_type == 'linear':
        vf = vfuncs.LinearValueFunction(**vf_params)
    elif args.vf_type == 'nn':
        vf = vfuncs.NnValueFunction(session=sess, ob_dim=ob_dim, **vf_params)

    # This is our parameter vector, though it won't be in the policy network.
    sy_logstd_a    = tf.get_variable("logstd", [ac_dim], initializer=tf.zeros_initializer())
    sy_oldlogstd_a = tf.placeholder(name="oldlogstd", shape=[ac_dim], dtype=tf.float32)

    # Set up some symbolic variables (i.e placeholders). Actions are now floats!
    sy_ob_no      = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    sy_ac_na      = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) 
    sy_adv_n      = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
    sy_h1         = utils.lrelu(utils.dense(sy_ob_no, 32, "h1", weight_init=utils.normc_initializer(1.0)))
    sy_h2         = utils.lrelu(utils.dense(sy_h1,    32, "h2", weight_init=utils.normc_initializer(1.0)))
    sy_mean_na    = utils.dense(sy_h2, ac_dim, "mean", weight_init=utils.normc_initializer(0.05))
    sy_oldmean_na = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)
    sy_n          = tf.shape(sy_ob_no)[0]

    # Make a batch version so that we get shape (n,a) or (1,a) and not (a,).
    sy_logstd_na    = tf.ones(shape=(sy_n,ac_dim), dtype=tf.float32) * sy_logstd_a
    sy_oldlogstd_na = tf.ones(shape=(sy_n,ac_dim), dtype=tf.float32) * sy_oldlogstd_a

    # Set up the Gaussian distribution stuff, plus diagnostics.
    sy_logprob_n  = utils.gauss_log_prob(mu=sy_mean_na, logstd=sy_logstd_na, x=sy_ac_na)
    sy_sampled_ac = (tf.random_normal(tf.shape(sy_mean_na)) * tf.exp(sy_logstd_na) + sy_mean_na)[0]
    sy_kl         = tf.reduce_mean(utils.gauss_KL(sy_mean_na, sy_logstd_na, sy_oldmean_na, sy_oldlogstd_na))
    sy_ent        = 0.5 * ac_dim * tf.log(2.*np.pi*np.e) + 0.5 * tf.reduce_sum(sy_logstd_a)

    # sy_surr: loss function that we'll differentiate to get the policy gradient
    sy_surr     = - tf.reduce_mean(sy_adv_n * sy_logprob_n) 
    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) 
    update_op   = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)
    
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101
    total_timesteps = 0
    stepsize = args.initial_stepsize

    for i in range(args.n_iter):
        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps.
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode = (len(paths) == 0 and (i%20 == 0) and args.render)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
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
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = utils.discount(rew_t, args.gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update and also re-fit the baseline.
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)

        # Policy update. I _think_ this is how we get the old logstd.
        _, surr_loss, oldmean_na, oldlogstd_a = sess.run(
                [update_op, sy_surr, sy_mean_na, sy_logstd_a], 
                feed_dict={sy_ob_no: ob_no, 
                           sy_ac_na: ac_n, 
                           sy_adv_n: standardized_adv_n, 
                           sy_stepsize: stepsize
                })

        kl, ent = sess.run([sy_kl, sy_ent],
                           feed_dict={sy_ob_no: ob_no, 
                                      sy_oldmean_na: oldmean_na,
                                      sy_oldlogstd_a: oldlogstd_a
                           })

        # Daniel: the rest of this for loop was provided in the starter code. I
        # assume for now that it goes _after_ all of the code we're writing.
        if kl > args.desired_kl * 2: 
            stepsize /= 1.5
            print('PG stepsize -> %s' % stepsize)
        elif kl < args.desired_kl / 2: 
            stepsize *= 1.5
            print('PG stepsize -> %s' % stepsize)
        else:
            print('PG stepsize OK')

        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([utils.pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", utils.explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", utils.explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("SurrogateLoss", surr_loss)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()
    tf.reset_default_graph()


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
        # HAVEN'T TESTED WITH NEW API
        vf = vfuncs.NnValueFunction(session=sess, ob_dim=ob_dim, **vf_params)

    if continuous_control:
        # HAVEN'T TESTED WITH NEW API
        ac_dim = env.action_space.shape[0]
        policyfn = policies.GaussianPolicy()
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

        # Policy update, plus diagnostics stuff.
        surr_loss, oldlogits_na = policyfn.update_policy(ob_no, ac_n, std_adv_n, stepsize)
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
