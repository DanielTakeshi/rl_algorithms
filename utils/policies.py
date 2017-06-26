"""
For managing policies. This seems like a better way to organize things. For now,
we have **stochastic** policies. This assumes the Python3 way of calling
superclasses' init methods.

TODO figure out a good way to integrate deterministic policies, figure out how
to get a good configuration file (for neural nets), etc. Lots of fun! :)

TODO figure out how to make assertions that we're in continuous vs discrete
spaces.

TODO have a net specification which we can use instead of hard-coding networks
here.
"""

import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
from . import utils_pg as utils


class StochasticPolicy(object):

    def __init__(self, sess, ob_dim, ac_dim):
        """ 
        Initializes the neural network policy. Right now there isn't much here,
        but this is a flexible design pattern for future versions of the code.
        """
        self.sess = sess

    def sample_action(self, x):
        """ To be implemented in the subclass. """
        raise NotImplementedError


class GibbsPolicy(StochasticPolicy):
    """ A policy where the action is to be sampled based on sampling a
    categorical random variable; this is for discrete control. """

    def __init__(self, sess, ob_dim, ac_dim):
        super().__init__(sess, ob_dim, ac_dim)

        # Placeholders for our inputs.
        self.ob_no = tf.placeholder(shape=[None, ob_dim], name="obs", dtype=tf.float32)
        self.ac_n  = tf.placeholder(shape=[None], name="act", dtype=tf.int32)
        self.adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        self.oldlogits_na = tf.placeholder(shape=[None, ac_dim], name='oldlogits', dtype=tf.float32)

        # Form the policy network and the log probabilities.
        self.hidden1 = layers.fully_connected(self.ob_no, 
                num_outputs=50,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=tf.nn.tanh)
        self.logits_na = layers.fully_connected(self.hidden1, 
                num_outputs=ac_dim,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=None)
        self.logp_na = tf.nn.log_softmax(self.logits_na)

        # Log probabilities of the actions in the minibatch, plus sampled action.
        self.nbatch     = tf.shape(self.ob_no)[0]
        self.logprob_n  = utils.fancy_slice_2d(self.logp_na, tf.range(self.nbatch), self.ac_n)
        self.sampled_ac = utils.categorical_sample_logits(self.logits_na)[0]

        # Policy gradients loss function and training step.
        self.surr_loss = - tf.reduce_mean(self.logprob_n * self.adv_n)
        self.stepsize  = tf.placeholder(shape=[], dtype=tf.float32)
        self.update_op = tf.train.AdamOptimizer(self.stepsize).minimize(self.surr_loss)

        # For KL divergence and entropy diagnostic purposes. These are computed
        # as averages across individual KL/entropy w.r.t each minibatch state.
        self.oldlogp_na = tf.nn.log_softmax(self.oldlogits_na)
        self.oldp_na    = tf.exp(self.oldlogp_na)
        self.p_na       = tf.exp(self.logp_na)
        self.kl_n       = tf.reduce_sum(self.oldp_na * (self.oldlogp_na - self.logp_na), axis=1)

        # I'm not sure why the KL divergence can be slightly negative. Each row
        # corresponds to a valid distribution. Must be numerical instability?
        self.assert_op  = tf.Assert(tf.reduce_all(self.kl_n >= -1e-4), [self.kl_n]) 
        with tf.control_dependencies([self.assert_op]):
            self.kl_n = tf.identity(self.kl_n)
        self.kl  = tf.reduce_mean(self.kl_n)
        self.ent = tf.reduce_mean(tf.reduce_sum( -self.p_na * self.logp_na, axis=1))


    def sample_action(self, ob):
        return self.sess.run(self.sampled_ac, feed_dict={self.ob_no: ob[None]})
 

    def update_policy(self, ob_no, ac_n, std_adv_n, stepsize):
        """ 
        Upon getting observations, those are fed through the network to get the
        logits. After this the computational graph eventually updates the
        policy, so the logits are now old logits. 
        """
        feed = {self.ob_no: ob_no,
                self.ac_n: ac_n,
                self.adv_n: std_adv_n,
                self.stepsize: stepsize}
        _, surr_loss, oldlogits_na = self.sess.run(
                [self.update_op, self.surr_loss, self.logits_na], feed_dict=feed)
        return surr_loss, oldlogits_na

       
    def kldiv_and_entropy(self, ob_no, oldlogits_na):
        """ Returning KL diverence and current entropy since they can re-use
        some of the computation. 
        
        In particular, the entropy doesn't need the old logits, since it just
        forwards the observation batch through the network to find the current
        log probabilities. """
        return self.sess.run([self.kl, self.ent], 
                feed_dict={self.ob_no:ob_no, self.oldlogits_na:oldlogits_na})


class GaussianPolicy(StochasticPolicy):
    """ A policy where the action is to be sampled based on sampling a Gaussian;
    this is for continuous control. """

    def __init__(self, sess, ob_dim, ac_dim):
        super().__init__(sess, ob_dim, ac_dim)

        # Placeholders for our inputs. Note that actions are floats.
        self.ob_no = tf.placeholder(shape=[None, ob_dim], name="obs", dtype=tf.float32)
        self.ac_na = tf.placeholder(shape=[None, ac_dim], name="act", dtype=tf.float32)
        self.adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)
        self.n     = tf.shape(self.ob_no)[0]
        
        # Special to the continuous case, the log std vector, it's a parameter.
        # Also, make batch versions so we get shape (n,a) (or (1,a)), not (a,).
        self.logstd_a     = tf.get_variable("logstd", [ac_dim], initializer=tf.zeros_initializer())
        self.oldlogstd_a  = tf.placeholder(name="oldlogstd", shape=[ac_dim], dtype=tf.float32)
        self.logstd_na    = tf.ones(shape=(self.n,ac_dim), dtype=tf.float32) * self.logstd_a
        self.oldlogstd_na = tf.ones(shape=(self.n,ac_dim), dtype=tf.float32) * self.oldlogstd_a

        # The policy network and the logits, which are the mean of a Gaussian.
        # Then don't forget to make an "old" version of that for KL divergences.
        self.hidden1 = layers.fully_connected(self.ob_no, 
                num_outputs=50,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=tf.nn.tanh)
        self.hidden2 = layers.fully_connected(self.hidden1, 
                num_outputs=50,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=tf.nn.tanh)
        self.mean_na = layers.fully_connected(self.hidden2, 
                num_outputs=ac_dim,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=None)
        self.oldmean_na = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32)

        # Diagonal Gaussian distribution for sampling actions and log probabilities.
        self.logprob_n  = utils.gauss_log_prob(mu=self.mean_na, logstd=self.logstd_na, x=self.ac_na)
        self.sampled_ac = (tf.random_normal(tf.shape(self.mean_na)) * tf.exp(self.logstd_na) + self.mean_na)[0]

        # Loss function that we'll differentiate to get the policy  gradient
        self.surr_loss = - tf.reduce_mean(self.logprob_n * self.adv_n) 
        self.stepsize  = tf.placeholder(shape=[], dtype=tf.float32) 
        self.update_op = tf.train.AdamOptimizer(self.stepsize).minimize(self.surr_loss)

        # KL divergence and entropy among Gaussian(s).
        self.kl  = tf.reduce_mean(utils.gauss_KL(self.mean_na, self.logstd_na, self.oldmean_na, self.oldlogstd_na))
        self.ent = 0.5 * ac_dim * tf.log(2.*np.pi*np.e) + 0.5 * tf.reduce_sum(self.logstd_a)


    def sample_action(self, ob):
        return self.sess.run(self.sampled_ac, feed_dict={self.ob_no: ob[None]})


    def update_policy(self, ob_no, ac_n, std_adv_n, stepsize):
        """ 
        The input is the same for the discrete control case, except we return a
        single log standard deviation vector in addition to our logits. In this
        case, the logits are really the mean vector of Gaussians, which differs
        among components (observations) in the minbatch. We return the *old*
        ones since they are assigned, then `self.update_op` runs, which makes
        them outdated.
        """
        feed = {self.ob_no: ob_no,
                self.ac_na: ac_n,
                self.adv_n: std_adv_n,
                self.stepsize: stepsize}
        _, surr_loss, oldmean_na, oldlogstd_a = self.sess.run(
                [self.update_op, self.surr_loss, self.mean_na, self.logstd_a],
                feed_dict=feed)
        return surr_loss, oldmean_na, oldlogstd_a

       
    def kldiv_and_entropy(self, ob_no, oldmean_na, oldlogstd_a):
        """ Returning KL diverence and current entropy since they can re-use
        some of the computation. For the KL divergence, though, we reuqire the
        old mean *and* the old log standard deviation to fully characterize the
        set of probability distributions we had earlier, each conditioned on
        different states in the MDP. Then we take the *average* of these, etc.,
        similar to the discrete case.
        """
        feed = {self.ob_no: ob_no,
                self.oldmean_na: oldmean_na,
                self.oldlogstd_a: oldlogstd_a}
        return self.sess.run([self.kl, self.ent], feed_dict=feed)
