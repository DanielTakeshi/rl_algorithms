"""
For managing policies. This seems like a better way to organize things. For now,
we have **stochastic** policies. This assumes the Python3 way of calling
superclasses' init methods.

TODO figure out a good way to integrate deterministic policies, figure out how
to get a good configuration file (for neural nets), etc. Lots of fun! :)
"""

import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
from . import utils_pg as utils


class StochasticPolicy(object):

    def __init__(self, sess, ob_dim, ac_dim):
        """ Initializes the neural network policy based on the `net_spec` and
        the current TensorFlow session `session`. """
        self.sess = sess

    def sample_action(self, x):
        """ To be implemented in the subclass. """
        raise NotImplementedError


class GibbsPolicy(StochasticPolicy):
    """ A policy where the action is to be sampled based on sampling a
    categorical random variable; this is for discrete control. """

    def __init__(self, sess, ob_dim, ac_dim):
        # TODO have a `net_spec` as input so we can specify the net in a better
        # way. Also see if there's a way we can dump this into the parent?
        #assert isinstance(obsfeat_space, Continuous) and \
        #        isinstance(action_space, Discrete)
        super().__init__(sess, ob_dim, ac_dim)

        # Various placeholders.
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

    def __init__(self, net_spec, session):
        assert isinstance(obsfeat_space, Continuous) and \
                isinstance(action_space, Continuous)
        super().__init__(self, net_spec, session)
        raise NotImplementedError

    def sample_action(self, meanvec):
        raise NotImplementedError
        pass
