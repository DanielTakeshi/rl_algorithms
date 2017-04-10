import numpy as np
np.set_printoptions(suppress=True, precision=5, edgeitems=10)
import tensorflow as tf
import tensorflow.contrib.distributions as distr
import gym
import logz
import scipy.signal
import sys


def gauss_log_prob(mu, logstd, x):
    """ Used for computing the log probability, following the formula for the
    multivariate Gaussian density. All the inputs should have shape (n,a). The
    `gp` contains the log probabilities for each of the dimensions of a to get
    (n,) as the result. This should generalize to a>1 but it doesn't generalize
    to non-diagonal covariance matrices.
    """
    var = tf.exp(2*logstd)
    gp = -tf.square(x - mu)/(2*var) - .5*tf.log(tf.constant(2*np.pi)) - logstd
    return tf.reduce_sum(gp, [1])


def gauss_KL(mu1, logstd1, mu2, logstd2, d):
    """ Returns KL divergence among two multivariate Gaussians, component-wise.
    Must assume diagonal matrix. All other inputs are shape (n,a). """
    var1 = tf.exp(2*logstd1)
    var2 = tf.exp(2*logstd2)
    kl_n = tf.reduce_sum(0.5*logstd2 - 0.5*logstd1 + (var1 + tf.square(mu1-mu2))/(2*var2) - 0.5*d, 
                         axis=[1]) 
    return kl_n


def normc_initializer(std=1.0):
    """ Initialize array with normalized columns """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def dense(x, size, name, weight_init=None):
    """ Dense (fully connected) layer """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b


def fancy_slice_2d(X, inds0, inds1):
    """ Like numpy's X[inds0, inds1] """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)


def discount(x, gamma):
    """
    Compute discounted sum of future values. Returns a list, NOT a scalar!
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]


def lrelu(x, leak=0.2):
    """ Performs a leaky ReLU operation. """
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y]. 
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1    
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories

    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the 
    probabilities of the different classes

    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)


def pathlength(path):
    return len(path["reward"])


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
        initialization so future calls to self.fit should remember this.
            
            sy_ytarg    (?,)
            sy_ob_no    (?,3)
            sy_h1       (?,32)
            sy_final_na (?,1)
            sy_ypred    (?,)
            sy_sq_diff  (?,)

        Edit: let's use the pre-processed version, with ob_dim*2+1 dimensions.
        """
        self.n_epochs    = n_epochs
        self.lrate       = stepsize
        self.sy_ytarg    = tf.placeholder(shape=[None], name="nnvf_y", dtype=tf.float32)
        self.sy_ob_no    = tf.placeholder(shape=[None, ob_dim*2+1], name="nnvf_ob", dtype=tf.float32)
        self.sy_h1       = lrelu(dense(self.sy_ob_no, 32, "nnvf_h1", weight_init=normc_initializer(1.0)), leak=0.0)
        self.sy_h2       = lrelu(dense(self.sy_h1, 32, "nnvf_h2", weight_init=normc_initializer(1.0)), leak=0.0)
        self.sy_final_na = dense(self.sy_h2, 1, "nnvf_final", weight_init=normc_initializer(1.0))
        self.sy_ypred    = tf.reshape(self.sy_final_na, [-1])
        self.sy_l2_error = tf.reduce_mean(tf.square(self.sy_ypred - self.sy_ytarg))
        self.fit_op      = tf.train.AdamOptimizer(stepsize).minimize(self.sy_l2_error)

        # Debugging. Then have a session here.
        print("\nself.sy_ytarg.shape = {}".format(self.sy_ytarg.get_shape()))
        print("self.sy_ob_no.shape = {}".format(self.sy_ob_no.get_shape()))
        print("self.sy_final_na.shape = {}".format(self.sy_final_na.get_shape()))
        print("self.sy_l2_error.shape = {}\n".format(self.sy_l2_error.get_shape()))
        self.sess = session

    def fit(self, X, y, session=None):
        """ 
        Updates weights (self.coef) with design matrix X (i.e. observations) and
        targets (i.e. actual returns) y.  I think we need a session?
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


def main_cartpole(n_iter=100, gamma=1.0, seed=0, min_timesteps_per_batch=1000, 
                  stepsize=1e-2, animate=True, vf_type='linear', vf_params=None, 
                  logdir=None):
    """ Runs vanilla policy gradient on the classic CartPole task.

    Symbolic variables have the prefix sy_, to distinguish them from the
    numerical values that are computed later in this function. Symbolic means
    that TF will not "compute values" until run in a session. Naming convention
    for shapes: `n` means batch size, `o` means observation dim, `a` means
    action dim. Also, some of these (e.g. sy_ob_no) are used both for when
    running the policy AND during training with a batch of observations.
    
      sy_ob_no:        batch of observations
      sy_ac_n:         batch of actions taken by the policy, for policy gradient computation
      sy_adv_n:        advantage function estimate
      sy_h1:           hidden layer (before this: input -> dense -> relu)
      sy_logits_na:    logits describing probability distribution of final layer
      sy_oldlogits_na: logits before updating, only for KL diagnostic
      sy_logp_na:      log probability of actions
      sy_sampled_ac:   sampled action when running the policy (NOT computing the policy gradient)
      sy_n:            clever way to obtain the batch size
      sy_logprob_n:    log-prob of actions taken -- used for policy gradient calculation

    Some of these rely on our convenience methods. Use a small initialization
    for the last layer, so the initial policy has maximal entropy. We are
    defaulting to a fully connected policy network with one hidden layer of 32
    units, and a softmax output (by default, applied to the last dimension,
    which we want here). Then we define a surrogate loss function. Again, it's
    the same as before, define a loss function and plug it into Adam.
   
    Args:
        n_iter: Number of iterations for policy gradient.
        gamma: The discount factor, used for computing returns.
        min_timesteps_per_batch: Minimum number of timesteps in a given
            iteration of policy gradients. Each trajectory consists of multiple
            timesteps.
        stepsize:
        animate: Whether to render it in OpenAI gym.
        logdir: Output directory for logging. If None, store to a random place.
    """
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make("CartPole-v0")
    ob_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    logz.configure_output_dir(logdir)

    # Create `sess` here so that we can pass it to the NN value function.
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)
    if vf_type == 'linear':
        vf = LinearValueFunction(**vf_params)
    elif vf_type == 'nn':
        vf = NnValueFunction(session=sess, ob_dim=ob_dim, **vf_params)

    # Symbolic variables as covered in the method documentation:
    sy_ob_no        = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    sy_ac_n         = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
    sy_adv_n        = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
    sy_h1           = lrelu(dense(sy_ob_no, 32, "h1", weight_init=normc_initializer(1.0)))
    sy_logits_na    = dense(sy_h1, num_actions, "final", weight_init=normc_initializer(0.05))
    sy_oldlogits_na = tf.placeholder(shape=[None, num_actions], name='oldlogits', dtype=tf.float32)
    sy_logp_na      = tf.nn.log_softmax(sy_logits_na)
    sy_sampled_ac   = categorical_sample_logits(sy_logits_na)[0]
    sy_n            = tf.shape(sy_ob_no)[0]
    sy_logprob_n    = fancy_slice_2d(sy_logp_na, tf.range(sy_n), sy_ac_n)

    # The following quantities are just used for computing KL and entropy, JUST FOR DIAGNOSTIC PURPOSES >>>>
    sy_oldlogp_na = tf.nn.log_softmax(sy_oldlogits_na)
    sy_oldp_na    = tf.exp(sy_oldlogp_na)
    sy_kl         = tf.reduce_sum(sy_oldp_na * (sy_oldlogp_na - sy_logp_na)) / tf.to_float(sy_n)
    sy_p_na       = tf.exp(sy_logp_na)
    sy_ent        = tf.reduce_sum( - sy_p_na * sy_logp_na) / tf.to_float(sy_n)
    # <<<<<<<<<<<<<

    # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")
    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n) 

    # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) 
    update_op = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)

    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101
    total_timesteps = 0

    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps.
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
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
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Estimate advantage function using baseline vf (these are lists!).
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
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

        # Policy update
        _, oldlogits_na = sess.run([update_op, sy_logits_na], 
                                   feed_dict={sy_ob_no:ob_no, 
                                              sy_ac_n:ac_n, 
                                              sy_adv_n:standardized_adv_n, 
                                              sy_stepsize:stepsize
                                   })
        kl, ent = sess.run([sy_kl, sy_ent], 
                           feed_dict={sy_ob_no:ob_no, 
                                      sy_oldlogits_na:oldlogits_na
                           })

        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()

    # Daniel: adding this to enable a for loop.
    tf.reset_default_graph()


def main_pendulum(logdir, seed, n_iter, gamma, min_timesteps_per_batch, 
                  initial_stepsize, desired_kl, vf_type, vf_params, animate=False):
    """ Runs policy gradients using a continuous action space now.

    Here are (some of) the symbolic variables I will be using. Convention for
    shaping is to use `n` (batch size), `o`, and/or `a` in the symbolic names.
    The following are for both the gradient and the policy:

      sy_ob_no:       batch of observations, both for PG computation AND running policy
      sy_h1/sy_h2:    both are hidden layers with leaky-relus.
      sy_mean_na:     final net output (like sy_logits_na from earlier), mean of a Gaussian
      sy_n:           clever way to obtain the batch size (or 1, for running policy)

    The following are for the policy, but not the gradient:

      sy_sampled_ac:  the current sampled action (a vector of controls) when running policy

    The following are for the gradient, but not the policy:

      sy_ac_na:       batch of actions taken by the policy, for PG computation
      sy_adv_n:       advantage function estimate (one per action vector)
      sy_logprob_n:   log-prob of actions taken in the batch, for PG computation

    Our parameters consists of (neural network weights, log std vector). The
    policy network will output the _mean_ of a Gaussian, NOT our actual action.
    Then the next set of parameters is the log std. Those two (the mean and log
    std) together define a distribution which we then sample from to get the
    actual action vector the agent plays. The net output, sy_mean_na, is a
    symbolic variable and thus NOT a parameter, but logstd_a IS a parameter.
    """

    tf.set_random_seed(seed)
    np.random.seed(seed)
    env = gym.make("Pendulum-v0")
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    logz.configure_output_dir(logdir)

    # Create `sess` here so that we can pass it to the NN value function.
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)
    if vf_type == 'linear':
        vf = LinearValueFunction(**vf_params)
    elif vf_type == 'nn':
        vf = NnValueFunction(session=sess, ob_dim=ob_dim, **vf_params)

    # This is still part of the parameters! I'm not setting it sy_ here.
    logstd_a       = tf.get_variable("logstdev", [ac_dim], initializer=tf.zeros_initializer())
    sy_oldlogstd_a = tf.placeholder(name="oldlogstdev", shape=[ac_dim], dtype=tf.float32)

    # Set up some symbolic variables (i.e placeholders). Actions are now floats!
    sy_ob_no      = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    sy_ac_na      = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) 
    sy_adv_n      = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
    sy_h1         = lrelu(dense(sy_ob_no, 32, "h1", weight_init=normc_initializer(1.0)))
    sy_h2         = lrelu(dense(sy_h1,    32, "h2", weight_init=normc_initializer(1.0)))
    sy_mean_na    = dense(sy_h2, ac_dim, "mean", weight_init=normc_initializer(0.05))
    sy_oldmean_na = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)
    sy_n          = tf.shape(sy_ob_no)[0]

    # Exponentiate and make a batch version so that we get shape (n,a) and not (a,).
    sy_logstd_na    = tf.ones(shape=(sy_n,ac_dim), dtype=tf.float32) * logstd_a
    sy_logoldstd_na = tf.ones(shape=(sy_n,ac_dim), dtype=tf.float32) * sy_oldlogstd_a

    # Set up the Gaussian distribution stuff, plus diagnostics.
    sy_logprob_n  = gauss_log_prob(sy_mean_na, sy_logstd_na, sy_ac_na)
    sy_sampled_ac = (tf.random_normal(tf.shape(sy_mean_na)) * tf.exp(sy_logstd_na) + sy_mean_na)[0]
    sy_kl         = tf.reduce_mean(gauss_KL(sy_mean_na, logstd_a, sy_oldmean_na, sy_oldlogstd_a, ac_dim))
    sy_ent        = 0.5 * ac_dim * tf.log(2.*np.pi*np.e) + 0.5 * tf.reduce_sum(logstd_a)

    # sy_surr: loss function that we'll differentiate to get the policy gradient
    sy_surr     = - tf.reduce_mean(sy_adv_n * sy_logprob_n) 
    sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) 
    update_op   = tf.train.AdamOptimizer(sy_stepsize).minimize(sy_surr)
    
    sess.__enter__() # equivalent to `with sess:`
    tf.global_variables_initializer().run() #pylint: disable=E1101
    total_timesteps = 0
    stepsize = initial_stepsize

    # Debugging.
    print("\nsy_ac_na.shape = {}".format(sy_ac_na.get_shape())) # (?,adim)
    print("sy_mean_na.shape = {}".format(sy_mean_na.get_shape())) # (?,adim)
    print("logstd_a.shape = {}".format(logstd_a.get_shape())) # (adim,)
    print("sy_logstd_na.shape = {}".format(sy_logstd_na.get_shape())) # (?,adim)
    print("sy_logprob_n.shape = {}".format(sy_logprob_n.get_shape())) # (?,)
    print("sy_sampled_ac.shape = {}".format(sy_sampled_ac.get_shape())) # (adim,)
    print("sy_kl.shape = {}".format(sy_kl.get_shape())) # ()
    print("sy_ent.shape = {}\n".format(sy_ent.get_shape())) # ()

    for i in range(n_iter):
        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps.
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode = (len(paths)==0 and (i%10 == 0) and animate)
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
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        # Estimate advantage function using baseline vf (these are lists!).
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
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
                [update_op, sy_mean_na, logstd_a], 
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
        if kl > desired_kl * 2: 
            stepsize /= 1.5
            print('stepsize -> %s'%stepsize)
        elif kl < desired_kl / 2: 
            stepsize *= 1.5
            print('stepsize -> %s'%stepsize)
        else:
            print('stepsize OK')

        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()

    # Daniel: adding this to enable a for loop.
    tf.reset_default_graph()


def main_cartpole1(d):
    return main_cartpole(**d)


def main_pendulum1(d):
    return main_pendulum(**d)


if __name__ == "__main__":
    """  This needs to be better-organized with a few arg-parses, etc. """

    if 0:
        # Part 0 (warm-up to ensure code is working)
        main_cartpole(logdir=None) # when you want to start collecting results, set the logdir
    if 0:
        # Part 1, just testing Pendulum.
        general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=500, initial_stepsize=1e-3)
        more_params = dict(logdir=None, seed=1, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params)
        main_pendulum(**more_params) 
    if 1:
        # Part 2, now comparing Cartpole with and without the neural network value function.
        head = 'outputs/cartpole/'
        general_params = dict(gamma=1.0, animate=False, min_timesteps_per_batch=1000, n_iter=100, stepsize=1e-2)
        params = [
            dict(logdir=head+'linearvf-seed0', seed=0, vf_type='linear', vf_params={}, **general_params),
            dict(logdir=head+'nnvf-seed0',     seed=0, vf_type='nn', vf_params=dict(n_epochs=50, stepsize=1e-3), **general_params),
            dict(logdir=head+'linearvf-seed1', seed=1, vf_type='linear', vf_params={}, **general_params),
            dict(logdir=head+'nnvf-seed1',     seed=1, vf_type='nn', vf_params=dict(n_epochs=50, stepsize=1e-3), **general_params),
            dict(logdir=head+'linearvf-seed2', seed=2, vf_type='linear', vf_params={}, **general_params),
            dict(logdir=head+'nnvf-seed2',     seed=2, vf_type='nn', vf_params=dict(n_epochs=50, stepsize=1e-3), **general_params),
        ]
        for p in params:
            main_cartpole1(p)
    if 0:
        # Part 2, now comparing Pendulum with and without the neural network value function.
        head = 'outputs/pendulum/'
        general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=500, initial_stepsize=1e-3)
        params = [
            dict(logdir=head+'linearvf-kl2e-3-seed0', seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            dict(logdir=head+'nnvf-kl2e-3-seed0',     seed=0, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=50, stepsize=1e-3), **general_params),
            dict(logdir=head+'linearvf-kl2e-3-seed1', seed=1, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            dict(logdir=head+'nnvf-kl2e-3-seed1',     seed=1, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=50, stepsize=1e-3), **general_params),
            dict(logdir=head+'linearvf-kl2e-3-seed2', seed=2, desired_kl=2e-3, vf_type='linear', vf_params={}, **general_params),
            dict(logdir=head+'nnvf-kl2e-3-seed2',     seed=2, desired_kl=2e-3, vf_type='nn', vf_params=dict(n_epochs=50, stepsize=1e-3), **general_params),
        ]
        # Actually, I can't get this work!
        #import multiprocessing
        #p = multiprocessing.Pool()
        #p.map(main_pendulum1, params)

        # Just do this instead, iterate through them.
        for p in params:
            main_pendulum1(p)
