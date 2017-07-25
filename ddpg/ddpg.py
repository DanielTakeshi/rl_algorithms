"""
Deep Deterministic Policy Gradients

Make Actor and Critic subclasses of a NNet class? Not sure ...  for now, I'll
put everything here but that might take a lot.
"""

import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
from replay_buffer import ReplayBuffer
sys.path.append("../")
from utils import logz


class DDPGAgent(object):

    def __init__(self, sess, env, args):
        self.sess = sess
        self.env = env
        self.args = args
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        # Placeholders for minibatches of data.
        self.obs_t_BO    = tf.placeholder(tf.float32, [None,self.ob_dim])
        self.act_t_BA    = tf.placeholder(tf.float32, [None,self.ac_dim])
        self.rew_t_B     = tf.placeholder(tf.float32, [None])
        self.obs_tp1_BO  = tf.placeholder(tf.float32, [None,self.ob_dim])
        self.done_mask_B = tf.placeholder(tf.float32, [None]) # end of episode = 1.

        # Construct the computational graphs and the experience replay buffer.
        self.actor   = Actor(sess, env, args, self.obs_t_BO)
        self.critic  = Critic(sess, env, args)
        self.rbuffer = ReplayBuffer(args.replay_size, self.ob_dim, self.ac_dim)

        self.sess.run(tf.global_variables_initializer())


    def train(self):
        """ 
        Algorithm 1 in the DDPG paper. 
        """
        num_episodes = 0
        t_start = time.time()
        obs = self.env.reset()

        for t in range(self.args.n_iter):
            print("\n*** DDPG Iteration {} ***".format(t))

            # Select actions and store this stuff in the replay buffer.
            act = self.sess.run(self.actor.actions_BA, {self.obs_t_BO: obs[None]})
            act = act[0]

            # Inject noise from our Gaussian process.
            # TODO

            # Now back to usual.
            new_obs, rew, done, info = self.env.step(act)
            self.rbuffer.add_sample(s=obs, a=act, r=rew, done=done)
            if done:
                obs = self.env.reset()
            else:
                obs = new_obs

            if (t > self.args.wait_until_rbuffer) and (t % self.args.learning_freq == 0):
                # Sample from the replay buffer.
                states_t_BO, actions_t_BA, rewards_t_B, states_tp1_BO, done_mask_B = \
                        self.rbuffer.sample(num=self.args.batch_size)

                feed = {self.obs_t_BO:    states_t_BO, 
                        self.act_t_BA:    actions_t_BA, 
                        self.rew_t_B:     rewards_t_B, 
                        self.obs_tp1_BO:  states_tp1_BO, 
                        self.done_mask_B: done_mask_B}

                # Update actor and critic networks?
                self.critic.update_weights()
                self.actor.update_weights()

                # Update target networks after some time?
                # TODO

            if (t % self.args.log_every_t_iter == 0):
                # Do some rollouts here.
                hours = (time.time()-t_start) / (60*60.)
                logz.log_tabular("TotalTimeHours",  hours)
                logz.log_tabular("TotalIterations", t)
                logz.dump_tabular()


    def test(self):
        """ """
        pass


class Actor(object):
    """ Given input as a batch of states, the actor deterministically provides
    us with actions, indicated as "mu" in the paper. 
    
    Since DDPG is off-policy, we can treat the problem of exploration
    independently from the learning algorithm. External to this class, I add
    Gaussian noise for this purpose.
    """

    def __init__(self, sess, env, args, obs_t_BO):
        self.sess = sess
        self.args = args
        self.ac_dim = env.action_space.shape[0]

        self.actions_BA = self._build_net(input_BO=obs_t_BO)
        #self.loss = ????????????
        #self.update_op = tf.train.AdamOptimizer(args.step_size_actor).minimize(self.loss)


    def _build_net(self, input_BO):
        """ The Actor network.
        
        Uses ReLUs for all hidden layers, but a tanh to the output to bound the
        action. This follows their 'low-dimensional networks' using 400 and 300
        units for the hidden layers.
        """
        hidden1 = layers.fully_connected(input_BO,
                num_outputs=400,
                weights_initializer=layers.xavier_initializer(),
                activation_fn=tf.nn.relu)
        hidden2 = layers.fully_connected(hidden1, 
                num_outputs=300,
                weights_initializer=layers.xavier_initializer(),
                activation_fn=tf.nn.relu)
        actions_BA = layers.fully_connected(hidden2,
                num_outputs=self.ac_dim,
                weights_initializer=layers.xavier_initializer(),
                activation_fn=tf.nn.tanh) # Note the tanh!
        # TODO multiply actions by a constant?
        return actions_BA

    
    def update_weights(self):
        # Run a session to execute `self.update_op`.
        pass


class Critic(object):
    """ Computes Q(s,a) values to encourage the Actor to learn better policies.

    This is colloquially referred to as 'Q' in the paper.
    """

    def __init__(self, sess, env, args):
        self.sess = sess
        self.args = args
        self.ac_dim = env.action_space.shape[0]

        #self.qvals_n = self._build_net(????????)
        #self.loss = ????????????
        #self.update_op = tf.train.AdamOptimizer(args.critic_step_size).minimize(self.loss)


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
