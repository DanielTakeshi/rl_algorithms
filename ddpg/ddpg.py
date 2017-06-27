"""
Deep Deterministic Policy Gradients

Make Actor and Critic subclasses of a NNet class? Not sure ...  for now, I'll
put everything here but that might take a lot.
"""

from replay_buffer import ReplayBuffer
import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers


class DDPGAgent(object):

    def __init__(self, sess, env, args):
        self.sess = sess
        self.env = env
        self.args = args

        # Construct the computational graphs and the experience replay buffer.
        self.actor   = Actor(sess, env, args)
        self.critic  = Critic(sess, args)
        self.rbuffer = ReplayBuffer(buffer_size=args.replay_size)


    def train(self):
        """ """
        # Write this method basically from Algorithm 1 in the DDPG paper.
        pass


    def test(self):
        """ """
        pass


class Actor(object):
    """ Given input as a batch of states, the actor deterministically provides
    us with the actions. 
    
    Since DDPG is off-policy, we can treat the problem of exploration
    independently from the learning algorithm. External to this class, I add
    Gaussian noise for this purpose.
    """

    def __init__(self, sess, env, args):
        self.sess = sess
        self.args = args
        self.ac_dim = env.action_space.shape[0]

        self.actions_na = self._build_net(??????????)
        self.loss = ????????????
        self.update_op = tf.train.AdamOptimizer(args.actor_step_size).minimize(self.loss)


    def _build_net(self, input_no):
        """ The Actor network.
        
        Uses ReLUs for all hidden layers, but a tanh to the output to bound the
        action. This follows their 'low-dimensional networks' using 400 and 300
        units for the hidden layers.
        """
        hidden1 = layers.fully_connected(input_no,
                num_outputs=400,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=tf.nn.relu)
        hidden2 = layers.fully_connected(hidden1, 
                num_outputs=300,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=tf.nn.relu)
        actions_na = layers.fully_connected(hidden2,
                num_outputs=self.ac_dim,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=tf.nn.tanh) # Note the tanh!
        return actions_na


    def update_weights(self):
        # Run a session to execute `self.update_op`.
        pass


class Critic(object):
    """ Computes Q(s,a) values to encourage the Actor to learn better policies.

    This is colloquially referred to as 'Q' in the paper.
    """

    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.ac_dim = ???????????

        self.qvals_n = self._build_net(????????)
        self.loss = ????????????
        self.update_op = tf.train.AdamOptimizer(args.critic_step_size).minimize(self.loss)


    def _build_net(self):
        """ The Critic network.
        
        Use ReLUs for all hidden layers. The actions are not included until
        **the second hidden layer**. TODO how do we do that? And what's the
        architecture here???
        """
        pass


    def update_weights(self):
        # Run a session to execute `self.update_op`.
        pass
