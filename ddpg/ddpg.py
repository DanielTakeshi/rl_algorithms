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
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        # Construct the networks and the experience replay buffer.
        self.actor   = Actor(sess, env, args)
        self.critic  = Critic(sess, env, args)
        self.rbuffer = ReplayBuffer(args.replay_size, self.ob_dim, self.ac_dim)

        self._debug_print()
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

            # Sample actions with noise injection and manage buffer.
            act = self.actor.sample_action(obs, train=True)
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


    def _debug_print(self):
        print("\n\t(A bunch of debug prints)\n")

        print("\nActor weights")
        for v in self.actor.weights:
            shp = v.get_shape().as_list()
            print("- {} shape:{} size:{}".format(v.name, shp, np.prod(shp)))
        print("Total # of weights: {}.".format(self.actor.num_weights))

        print("\nCritic weights")
        for v in self.critic.weights:
            shp = v.get_shape().as_list()
            print("- {} shape:{} size:{}".format(v.name, shp, np.prod(shp)))
        print("Total # of weights: {}.".format(self.critic.num_weights))



class Network(object):
    """ 
    Just so the Actor and Critic nets don't have a lot of duplicate code. This
    way they can refer to the similar sets of placeholders (but not the exact
    same ones in memory, just a copy) and I can change it easily here.
    """

    def __init__(self, sess, env, args):
        self.sess = sess
        self.args = args

        # Some random stuff.
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]
        self.ac_high = env.action_space.high
        self.ac_low = env.action_space.low

        # Placeholders for minibatches of data. End of episode = 1 for mask.
        self.obs_t_BO    = tf.placeholder(tf.float32, [None,self.ob_dim])
        self.act_t_BA    = tf.placeholder(tf.float32, [None,self.ac_dim])
        self.rew_t_B     = tf.placeholder(tf.float32, [None])
        self.obs_tp1_BO  = tf.placeholder(tf.float32, [None,self.ob_dim])
        self.done_mask_B = tf.placeholder(tf.float32, [None])



class Actor(Network):
    """ Given input as a batch of states, the actor deterministically provides
    us with actions, indicated as "mu" in the paper. 
    
    Since DDPG is off-policy, we can treat the problem of exploration
    independently from the learning algorithm. External to this class, I add
    Gaussian noise for this purpose.
    """

    def __init__(self, sess, env, args):
        super().__init__(sess, env, args)

        # The action network and its corresponding taget.
        self.actions_BA      = self._build_net(self.obs_t_BO, scope='ActorNet')
        self.actions_targ_BA = self._build_net(self.obs_t_BO, scope='TargActorNet')

        # Collect weights since it's generally convenient to do so.
        self.weights      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ActorNet')
        self.weights_targ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='TargActorNet')
        self.weights_v      = tf.concat([tf.reshape(w, [-1]) for w in self.weights], axis=0)
        self.weights_v_targ = tf.concat([tf.reshape(w, [-1]) for w in self.weights_targ], axis=0)

        # These should be the same among both nets.
        self.w_shapes = [w.get_shape().as_list() for w in self.weights]
        self.num_weights = np.sum([np.prod(sh) for sh in self.w_shapes])


    def _build_net(self, input_BO, scope):
        """ The Actor network.
        
        Uses ReLUs for all hidden layers, but a tanh to the output to bound the
        action. This follows their 'low-dimensional networks' using 400 and 300
        units for the hidden layers. Set `reuse=False`. I don't use batch
        normalization or their precise weight initialization.
        """
        with tf.variable_scope(scope, reuse=False):
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
            # This should broadcast, but haven't tested with ac_dim > 1.
            actions_BA = tf.multiply(actions_BA, self.ac_high)
            return actions_BA


    def sample_action(self, obs, train=True):
        """ Samples an action.
        
        TODO we don't have their exact Gaussian noise injection process because
        I can't figure out how to implement it. :-(

        Parameters
        ----------
        obs: [np.array]
            Represents current states. We assume we need to expand it.
        train: [boolean]
            True means we need to inject noise. False is for test evaluation.
        """
        act = self.sess.run(self.actions_BA, {self.obs_t_BO: obs[None]})
        act = act[0]
        if train:
            return act + np.random.normal(loc=self.args.ou_noise_theta,
                    scale=self.args.ou_noise_sigma, size=act.shape)
        else:
            return act

    
    def update_weights(self):
        # Run a session to execute `self.update_op`.
        pass



class Critic(Network):
    """ Computes Q(s,a) values to encourage the Actor to learn better policies.

    This is colloquially referred to as 'Q' in the paper.
    """

    def __init__(self, sess, env, args):
        super().__init__(sess, env, args)

        # The critic network (i.e. Q-values) and its corresponding target.
        self.qvals_B      = self._build_net(self.obs_t_BO, self.act_t_BA, scope='CriticNet')
        self.qvals_targ_B = self._build_net(self.obs_t_BO, self.act_t_BA, scope='TargCriticNet')

        # Collect weights since it's generally convenient to do so.
        self.weights      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='CriticNet')
        self.weights_targ = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='TargCriticNet')
        self.weights_v      = tf.concat([tf.reshape(w, [-1]) for w in self.weights], axis=0)
        self.weights_v_targ = tf.concat([tf.reshape(w, [-1]) for w in self.weights_targ], axis=0)

        # These should be the same among both nets.
        self.w_shapes = [w.get_shape().as_list() for w in self.weights]
        self.num_weights = np.sum([np.prod(sh) for sh in self.w_shapes])


    def _build_net(self, input_BO, acts_BO, scope):
        """ The critic network.
        
        Use ReLUs for all hidden layers. The actions are not included until
        **the second hidden layer**. The output consists of one Q-value for each
        batch. Set `reuse=False`. I don't use batch normalization or their
        precise weight initialization.
        """
        with tf.variable_scope(scope, reuse=False):
            hidden1 = layers.fully_connected(input_BO,
                    num_outputs=400,
                    weights_initializer=layers.xavier_initializer(),
                    activation_fn=tf.nn.relu)
            # Insert the concatenation here. This should be fine, I think.
            state_action = tf.concat(axis=1, values=[hidden1, acts_BO])
            hidden2 = layers.fully_connected(state_action,
                    num_outputs=300,
                    weights_initializer=layers.xavier_initializer(),
                    activation_fn=tf.nn.relu)
            qvals_B = layers.fully_connected(hidden2,
                    num_outputs=1,
                    weights_initializer=layers.xavier_initializer(),
                    activation_fn=None)
            return qvals_B


    def update_weights(self):
        # Run a session to execute `self.update_op`.
        pass
