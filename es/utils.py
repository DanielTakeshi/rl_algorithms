"""
A file for utility methods, to reduce clutter in `gail.py`.

(c) April 2017 by Daniel Seita
"""

import numpy as np
import tensorflow as tf


def get_tf_session():
    """ Returning a session. Set options here (e.g. for GPUs) if desired. """
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1,
                               intra_op_parallelism_threads=1)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def get_available_gpus():
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

    print("AVAILABLE GPUS: ", get_available_gpus())
    return session


def get_batch(expert_obs, expert_act, batch_size):
    """ Obtain a minibatch of samples. """
    indices = np.arange(expert_obs.shape[0])
    np.random.shuffle(indices)
    xs = expert_obs[indices[:batch_size]]
    ys = expert_act[indices[:batch_size]]
    return xs, ys


def load_data(args):
    """ Load the expert trajectories and return observations, actions, and
    a combined shape. """
    str_roll = str(args.expert_rollouts).zfill(4)
    expert_data = np.load('data/'+args.envname+'_'+str_roll+'.npy')
    expert_obs = expert_data[()]['observations']
    expert_act = np.squeeze(expert_data[()]['actions'])
    N = expert_obs.shape[0]
    assert N == expert_act.shape[0]
    combo_shape = []
    obs_shape = list(expert_obs.shape)[1:]
    act_shape = list(expert_act.shape)[1:]
    for i in range(len(obs_shape)):
        combo_shape.append(obs_shape[i]+act_shape[i])
    print("\n(traj) expert_obs.shape = {}".format(expert_obs.shape))
    print("(traj) expert_act.shape = {}".format(expert_act.shape))
    print("combo_shape = {}".format(combo_shape))
    return expert_obs, expert_act, obs_shape, act_shape, combo_shape


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
    kl_n = tf.reduce_sum(logstd2 - logstd1 + (var1 + tf.square(mu1-mu2))/(2*var2) - 0.5*d, 
                         axis=[1]) 
    return kl_n


def normc_initializer(std=1.0):
    """ Initialize array with normalized columns """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


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
