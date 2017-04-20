"""
Trust Region Policy Optimization

For usage, to quickly test, use:

    python main.py Pendulum-v0 --vf_type nn --do_not_save --render

(c) April 2017 by Daniel Seita. This code is built upon starter code from
Berkeley CS 294-112.
"""

import argparse
import numpy as np
np.set_printoptions(suppress=True, precision=5, edgeitems=10)
import tensorflow as tf
import tensorflow.contrib.distributions as distr
import gym
import sys
if "../" not in sys.path:
    sys.path.append("../")
from utils import utils_pg as utils
from utils import logz


class LinearValueFunction(object):
    """ Estimates the baseline function for PGs via ridge regression. """
    coef = None

    def fit(self, X, y):
        """ 
        Updates weights (self.coef) with design matrix X (i.e. observations) and
        targets (i.e. actual returns) y. 
        """
        assert X.shape[0] == y.shape[0]
        assert len(y.shape) == 1
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)

    def predict(self, X):
        """ Predicts return from observations (i.e. environment states) X. """
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)

    def preproc(self, X):
        """ Adding a bias column, and also adding squared values (huh). """
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)


class NnValueFunction(object):
    """ Estimates the baseline function for PGs via neural network. """

    def __init__(self, session, ob_dim=None, n_epochs=10, stepsize=1e-3):
        """ 
        They provide us with an ob_dim in the code so I assume we can use it;
        makes it easy to define the layers anyway. This gets constructed upon
        initialization so future calls to self.fit should remember this. I
        actually use the pre-processed version, though.
        """
        self.n_epochs    = n_epochs
        self.lrate       = stepsize
        self.sy_ytarg    = tf.placeholder(shape=[None], name="nnvf_y", dtype=tf.float32)
        self.sy_ob_no    = tf.placeholder(shape=[None, ob_dim*2+1], name="nnvf_ob", dtype=tf.float32)
        self.sy_h1       = utils.lrelu(utils.dense(self.sy_ob_no, 32, "nnvf_h1", weight_init=utils.normc_initializer(1.0)), leak=0.0)
        self.sy_h2       = utils.lrelu(utils.dense(self.sy_h1, 32, "nnvf_h2", weight_init=utils.normc_initializer(1.0)), leak=0.0)
        self.sy_final_na = utils.dense(self.sy_h2, 1, "nnvf_final", weight_init=utils.normc_initializer(1.0))
        self.sy_ypred    = tf.reshape(self.sy_final_na, [-1])
        self.sy_l2_error = tf.reduce_mean(tf.square(self.sy_ypred - self.sy_ytarg))
        self.fit_op      = tf.train.AdamOptimizer(stepsize).minimize(self.sy_l2_error)
        self.sess = session

    def fit(self, X, y, session=None):
        """ Updates weights (self.coef) with design matrix X (i.e. observations)
        and targets (i.e. actual returns) y.
        """
        assert X.shape[0] == y.shape[0]
        assert len(y.shape) == 1
        Xp = self.preproc(X)
        for i in range(self.n_epochs):
            _,err = self.sess.run(
                    [self.fit_op, self.sy_l2_error], 
                    feed_dict={self.sy_ob_no: Xp,
                               self.sy_ytarg: y
                    })

    def predict(self, X):
        """ 
        Predicts returns from observations (i.e. environment states) X. I also
        think we need a session here. No need to expand dimensions, BTW! It's
        effectively already done for us elsewhere.
        """
        Xp = self.preproc(X)
        return self.sess.run(self.sy_ypred, feed_dict={self.sy_ob_no:Xp})

    def preproc(self, X):
        """ Let's add this here to increase dimensionality. """
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)


def trpo_continuous(logdir, args, vf_params):
    """ Runs TRPO on environments with continuous action spaces.

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
        vf = LinearValueFunction(**vf_params)
    elif args.vf_type == 'nn':
        vf = NnValueFunction(session=sess, ob_dim=ob_dim, **vf_params)

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

    # Set up the Gaussian distribution stuff & diagnostics. New in TRPO: old logprob.
    sy_logprob_n     = utils.gauss_log_prob(mu=sy_mean_na, logstd=sy_logstd_na, x=sy_ac_na)
    sy_oldlogprob_n  = utils.gauss_log_prob(mu=sy_oldmean_na, logstd=sy_oldlogstd_na, x=sy_ac_na)
    sy_sampled_ac    = (tf.random_normal(tf.shape(sy_mean_na)) * tf.exp(sy_logstd_na) + sy_mean_na)[0]
    sy_kl            = tf.reduce_mean(utils.gauss_KL(sy_mean_na, sy_logstd_na, sy_oldmean_na, sy_oldlogstd_na))
    sy_ent           = 0.5 * ac_dim * tf.log(2.*np.pi*np.e) + 0.5 * tf.reduce_sum(sy_logstd_a)

    # sy_surr: this time, we have a ratio exp(logprob/oldlogprob) times advangage.
    sy_surr     = - tf.reduce_mean(sy_adv_n * tf.exp(sy_logprob_n - sy_oldlogprob_n))
    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) 

    # TODO I don't think we can apply this directly. We should get the gradient
    # based on sy_surr, but then we need to rescale by the conjugate gradient
    # and *then* do a line search.
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
        _, oldmean_na, oldlogstd_a = sess.run(
                [update_op, sy_mean_na, sy_logstd_a], 
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
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()
    tf.reset_default_graph()


if __name__ == "__main__":
    """ 
    This needs to be better-organized but it will do for now. For usage, see the
    top of this file. For now, assume we only care about environments with
    continuous actions.
    """
    p = argparse.ArgumentParser()
    p.add_argument('envname', type=str)
    p.add_argument('--render', action='store_true')
    p.add_argument('--do_not_save', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--desired_kl', type=float, default=2e-3)
    p.add_argument('--gamma', type=float, default=0.97)
    p.add_argument('--min_timesteps_per_batch', type=int, default=2500) 
    p.add_argument('--n_iter', type=int, default=500)
    p.add_argument('--initial_stepsize', type=float, default=1e-3)
    p.add_argument('--vf_type', type=str, default='linear')
    p.add_argument('--nnvf_epochs', type=int, default=50)
    p.add_argument('--nnvf_ssize', type=float, default=1e-3)
    args = p.parse_args()

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

    trpo_continuous(logdir, args, vf_params)
