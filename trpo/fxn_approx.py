"""
This will make some function approximators that we can use, particularly: linear
and neural network value functions. Instantiate instances of these in other
pieces of the code base.

(c) April 2017 by Daniel Seita, built upon `starter code` from John Schulman.
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as distr
import sys
if "../" not in sys.path:
    sys.path.append("../")
from utils import utils_pg as utils


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
        self.sy_ob_no    = tf.placeholder(shape=[None, ob_dim+1], name="nnvf_ob", dtype=tf.float32)
        self.sy_h1       = utils.lrelu(utils.dense(self.sy_ob_no, 32, "nnvf_h1", weight_init=utils.normc_initializer(1.0)), leak=0.0)
        self.sy_h2       = utils.lrelu(utils.dense(self.sy_h1, 32, "nnvf_h2", weight_init=utils.normc_initializer(1.0)), leak=0.0)
        self.sy_final_n  = utils.dense(self.sy_h2, 1, "nnvf_final", weight_init=utils.normc_initializer(1.0))
        self.sy_ypred    = tf.reshape(self.sy_final_n, [-1])
        self.sy_l2_error = tf.reduce_mean(tf.square(self.sy_ypred - self.sy_ytarg))
        self.fit_op      = tf.train.AdamOptimizer(stepsize).minimize(self.sy_l2_error)
        self.sess = session

    def fit(self, X, y):
        """ Updates weights (self.coef) with design matrix X (i.e. observations)
        and targets (i.e. actual returns) y. NOTE! We now return a dictionary
        `out` so that we can provide information relevant information for the
        logger.
        """
        assert X.shape[0] == y.shape[0]
        assert len(y.shape) == 1
        out = {}
        out["PredStdevBefore"] = self.predict(X).std()

        Xp = self.preproc(X)
        for i in range(self.n_epochs):
            _,err = self.sess.run(
                    [self.fit_op, self.sy_l2_error], 
                    feed_dict={self.sy_ob_no: Xp,
                               self.sy_ytarg: y
                    })

        out["PredStdevAfter"] = self.predict(X).std()
        out["TargStdev"] = y.std()
        return out

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
        #return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)
        return np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
