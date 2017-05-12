"""
This is Natural Evolution Strategies, designed to run on one computer and not a
cluster.

(c) May 2017 by Daniel Seita, though obviously based on OpenAI's work/idea.
"""

import gym
import logz
import numpy as np
import pickle
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
import utils
from collections import defaultdict
np.set_printoptions(edgeitems=100, linewidth=100, suppress=True, precision=5)


class ESAgent:

    def __init__(self, session, args, log_dir=None, continuous=True):
        """ An Evolution Strategies agent.

        It uses the same network architecture from OpenAI's paper for sampling
        actions. The agent has functionality for obtaining and updating weights
        in vector form to make ES addition easier.

        Args:
            session: A Tensorflow session.
            args: The argparse from the user.
            log_dir: The log directory for the logging, if any.
            continuous: Whether the agent acts in a continuous or discrete
                action space. (Right now only continuous is supported.)
        """
        assert continuous == True, "Error: only continuous==True is supported."
        self.sess = session
        self.args = args
        self.log_dir = log_dir
        self.env = gym.make(args.envname)
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.shape[0]
        self.ob_no = tf.placeholder(shape=[None, ob_dim], dtype=tf.float32)

        # Build the final network layer and perform action sampling.
        self.net_final_layer, self.logstd_a = \
                self._make_network(data_in=self.ob_no, out_dim=ac_dim)
        self.sampled_ac = (tf.random_normal(tf.shape(self.net_final_layer)) * \
                tf.exp(self.logstd_a) + self.net_final_layer)[0]
 
        # To *extract* weight values, run a session on `self.weights_v`.
        self.weights   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ESAgent')
        self.weights_v = tf.concat([tf.reshape(w, [-1]) for w in self.weights], axis=0)
        self.shapes    = [w.get_shape().as_list() for w in self.weights]
        self.num_ws    = np.sum([np.prod(sh) for sh in self.shapes])

        # To *update* weights, run `self.set_params_op` w/feed `self.new_weights_v`.
        self.new_weights_v = tf.placeholder(tf.float32, shape=[self.num_ws])
        updates = []
        start = 0
        for (i,w) in enumerate(self.weights):
            shape = self.shapes[i]
            size = np.prod(shape)
            updates.append(
                    tf.assign(w, tf.reshape(self.new_weights_v[start:start+size], shape))
            )
            start += size
        self.set_params_op = tf.group(*updates)

        if args.verbose:
            self._print_summary()
        self.sess.run(tf.global_variables_initializer())
        logz.configure_output_dir(log_dir)


    def _make_network(self, data_in, out_dim):
        """ Build the network with the same architecture following OpenAI's paper.

        Returns the final *layer* of the network and the log std for continuous
        environments. To sample an action, one needs an extra sampling layer. We
        don't use a non-linearity for the last layer because different envs have
        different action ranges.
        """
        with tf.variable_scope("ESAgent", reuse=False):
            out = data_in
            out = layers.fully_connected(out, num_outputs=64,
                    weights_initializer = layers.xavier_initializer(uniform=True),
                    #weights_initializer = utils.normc_initializer(0.5),
                    activation_fn = tf.nn.tanh)
            out = layers.fully_connected(out, num_outputs=64,
                    weights_initializer = layers.xavier_initializer(uniform=True),
                    #weights_initializer = utils.normc_initializer(0.5),
                    activation_fn = tf.nn.tanh)
            out = layers.fully_connected(out, num_outputs=out_dim,
                    weights_initializer = layers.xavier_initializer(uniform=True),
                    #weights_initializer = utils.normc_initializer(0.5),
                    activation_fn = None)
            logstd_a = tf.get_variable("logstd", [out_dim], 
                    initializer=tf.constant_initializer(-1.0))
        return out, logstd_a


    def _compute_return(self, test=False):
        """ Runs the current neural network policy. 

        For now, we assume we run one episode. Also, we expand the observations
        to get a dummy dimension, in case we figure out how to make use of
        minibatches later.
        
        Args:
            test True if testing, False if part of training.

        Returns:
            The scalar return to be evaluated by the ES agent.
        """
        max_steps = self.env.spec.timestep_limit
        obs = self.env.reset()
        done = False
        steps = 0
        total_rew = 0

        while not done:
            exp_obs = np.expand_dims(obs, axis=0)
            action = self.sess.run(self.sampled_ac, {self.ob_no:exp_obs})
            obs, r, done, _ = self.env.step(action)
            total_rew += r
            steps += 1
            if self.args.render and test:
                self.env.render()
            if steps >= max_steps or done:
                break

        return total_rew


    def _print_summary(self):
        """ Just for debugging assistance. """
        print("\nES Agent NN weight shapes:\n{}".format(self.shapes))
        print("\nES Agent NN weights:")
        for w in self.weights:
            print(w)
        print("\nNumber of weights: {}".format(self.num_ws))
        print("\naction space: {}".format(self.env.action_space))
        print("lower bound: {}".format(self.env.action_space.low))
        print("upper bound: {}\n".format(self.env.action_space.high))


    def run_es(self):
        """ Runs Evolution Strategies.

        Tricks used:
            - Antithetic (i.e. mirrored) sampling 

        The final weights are saved and can be pre-loaded elsewhere.
        """
        args = self.args
        t_start = time.time()

        for i in range(args.es_iters):
            if (i % args.log_every_t_iter == 0):
                print("\n************ Iteration %i ************"%i)
            stats = defaultdict(list)

            # Set stuff up for perturbing weights and determining fitness.
            weights_old = self.sess.run(self.weights_v)
            tmp = np.random.randn(args.npop / 2, self.num_ws)
            N = np.concatenate((tmp, -tmp), axis=0)
            scores = []

            for j in range(args.npop):
                weights_new = weights_old + args.sigma * N[j]
                self.sess.run(self.set_params_op, 
                              feed_dict={self.new_weights_v: weights_new})
                scores.append(self._compute_return())

            # Determine the new weights based on the scores using a weighted
            # update. F.shape=(npop,1), N.shape=(npop,num_weights).
            F = (scores - np.mean(scores)) / (np.std(scores)+1e-8)
            alpha = (args.lrate_es / (args.sigma*args.npop))
            next_weights = weights_old + alpha * np.dot(N.T, F)
            self.sess.run(self.set_params_op, 
                          feed_dict={self.new_weights_v: next_weights})

            # Test roll-outs with these new weights.
            returns = []
            for _ in range(args.test_trajs):
                returns.append(self._compute_return(test=True))
            
            # Report relevant logs.
            if (i % args.log_every_t_iter == 0):
                minutes = (time.time()-t_start) / 60.
                logz.log_tabular("ScoresAvg",        np.mean(scores))
                logz.log_tabular("ScoresStd",        np.std(scores))
                logz.log_tabular("ScoresMax",        np.max(scores))
                logz.log_tabular("ScoresMin",        np.min(scores))
                logz.log_tabular("FinalAvgReturns",  np.mean(returns))
                logz.log_tabular("FinalStdReturns",  np.std(returns))
                logz.log_tabular("FinalMaxReturns",  np.max(returns))
                logz.log_tabular("FinalMinReturns",  np.min(returns))
                logz.log_tabular("TotalTimeMinutes", minutes)
                logz.log_tabular("TotalIterations",  i)
                logz.dump_tabular()

        # Save the policy so I can test it later.
        #TODO
