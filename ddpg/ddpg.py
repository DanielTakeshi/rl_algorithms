"""
Deep Deterministic Policy Gradients

Make Actor and Critic subclasses of a NNet class? Not sure ...  for now, I'll
put everything here but that might take a lot.
"""

import gym
import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
from replay_buffer import ReplayBuffer
from collections import defaultdict
sys.path.append("../")
from utils import logz


class DDPGAgent(object):

    def __init__(self, sess, env, test_env, args):
        self.sess = sess
        self.args = args
        self.env = env
        self.test_env = test_env
        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        # Construct the networks and the experience replay buffer.
        self.actor   = Actor(sess, env, args)
        self.critic  = Critic(sess, env, args)
        self.rbuffer = ReplayBuffer(args.replay_size, self.ob_dim, self.ac_dim)

        # Initialize then run, also setting current=target to start.
        self._debug_print()
        self.sess.run(tf.global_variables_initializer())
        self.actor.update_target_net(smooth=False)
        self.critic.update_target_net(smooth=False)


    def train(self):
        """ 
        Algorithm 1 in the DDPG paper. 
        """
        num_episodes = 0
        t_start = time.time()
        obs = self.env.reset()

        for t in range(self.args.n_iter):
            if (t % self.args.log_every_t_iter == 0) and (t > self.args.wait_until_rbuffer):
                print("\n*** DDPG Iteration {} ***".format(t))

            # Sample actions with noise injection and manage buffer.
            act = self.actor.sample_action(obs, train=True)
            new_obs, rew, done, info = self.env.step(act)
            self.rbuffer.add_sample(s=obs, a=act, r=rew, done=done)
            if done:
                obs = self.env.reset()
                num_episodes += 1
            else:
                obs = new_obs

            if (t > self.args.wait_until_rbuffer) and (t % self.args.learning_freq == 0):
                # Sample from the replay buffer.
                states_t_BO, actions_t_BA, rewards_t_B, states_tp1_BO, done_mask_B = \
                        self.rbuffer.sample(num=self.args.batch_size)

                feed = {'obs_t_BO':    states_t_BO, 
                        'act_t_BA':    actions_t_BA, 
                        'rew_t_B':     rewards_t_B, 
                        'obs_tp1_BO':  states_tp1_BO, 
                        'done_mask_B': done_mask_B}

                # Update the critic, get sampled policy gradients, update actor.
                gradients = self.critic.update_weights(feed)
                self.actor.update_weights(feed, gradients)

                # Update both target networks.
                self.critic.update_target_net()
                self.actor.update_target_net()

            if (t % self.args.log_every_t_iter == 0) and (t > self.args.wait_until_rbuffer):
                # Do some rollouts here and then record statistics.
                stats = self._do_rollouts()
                hours = (time.time()-t_start) / (60*60.)
                logz.log_tabular("MeanReward",      np.mean(stats['reward']))
                logz.log_tabular("MaxReward",       np.max(stats['reward']))
                logz.log_tabular("MinReward",       np.min(stats['reward']))
                logz.log_tabular("StdReward",       np.std(stats['reward']))
                logz.log_tabular("MeanLength",      np.mean(stats['length']))
                logz.log_tabular("NumTrainingEps",  num_episodes)
                logz.log_tabular("TotalTimeHours",  hours)
                logz.log_tabular("TotalIterations", t)
                logz.dump_tabular()


    def _do_rollouts(self):
        """ 
        Some rollouts to evaluate the agent's progress.  Returns a dictionary
        containing relevant statistics. 
        """
        num_episodes = 50
        stats = defaultdict(list)

        for i in range(num_episodes):
            obs = self.test_env.reset()
            ep_time = 0
            ep_reward = 0

            # Run one episode ...
            while True:
                act = self.actor.sample_action(obs, train=False)
                new_obs, rew, done, info = self.test_env.step(act)
                ep_time += 1
                ep_reward += rew
                if done:
                    break

            # ... and collect its information here.
            stats['length'].append(ep_time)
            stats['reward'].append(ep_reward)

        return stats


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
    Just so the Actor and Critic nets don't have more duplicate code. This way
    they can refer to the similar sets of placeholders (but not the exact same
    ones in memory, just a copy) and I can change it easily here.
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

        # Update the target action network. Provide hard and smooth updates.
        target_smooth = []
        target_hard = []
        for var, var_target in zip(sorted(self.weights,      key=lambda v: v.name),
                                   sorted(self.weights_targ, key=lambda v: v.name)):
            update_sm = self.args.tau * var + (1 - self.args.tau) * var_target
            target_smooth.append(var_target.assign(update_sm))
            target_hard.append(var_target.assign(var))
        self.update_target_smooth = tf.group(*target_smooth)
        self.update_target_hard   = tf.group(*target_hard)

        # The Actor _update_, with gradients provided by the Critic. TODO check!
        self.act_grads_BA = tf.placeholder(tf.float32, [None,self.ac_dim])
        self.actor_gradients = tf.gradients(self.actions_BA, self.weights, -self.act_grads_BA)
        self.optimize = tf.train.AdamOptimizer(self.args.step_size_actor).\
                    apply_gradients(zip(self.actor_gradients, self.weights))


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

    
    def update_target_net(self, smooth=True):
        """ 
        Update the target network based on the current weights. Normally we do
        this with smooth=True except for the first step, or unless we want to
        see how poorly hard updates perform generally.
        """
        if smooth:
            self.sess.run(self.update_target_smooth)
        else:
            self.sess.run(self.update_target_hard)


    def update_weights(self, f, gradients):
        """ Gradient-based update of current actor parameters. """
        feed = {
            self.obs_t_BO:     f['obs_t_BO'],
            self.act_t_BA:     f['act_t_BA'],
            self.rew_t_B:      f['rew_t_B'],
            self.obs_tp1_BO:   f['obs_tp1_BO'],
            self.done_mask_B:  f['done_mask_B'],
            self.act_grads_BA: gradients[0] # TODO check ..
        }
        self.sess.run(self.optimize, feed)



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

        # Update the target action network. Provide hard and smooth updates.
        target_smooth = []
        target_hard = []
        for var, var_target in zip(sorted(self.weights,      key=lambda v: v.name),
                                   sorted(self.weights_targ, key=lambda v: v.name)):
            update_sm = self.args.tau * var + (1 - self.args.tau) * var_target
            target_smooth.append(var_target.assign(update_sm))
            target_hard.append(var_target.assign(var))
        self.update_target_smooth = tf.group(*target_smooth)
        self.update_target_hard   = tf.group(*target_hard)

        # The _critic_ uses y_i, the target for its loss. Depends on `done` mask! 
        self.target_val_B = self.rew_t_B + (self.args.Q_gamma * self.qvals_targ_B) * (1 - self.done_mask_B)
        self.l2_error = tf.reduce_mean(tf.square(self.target_val_B - self.qvals_B))
        # TODO l2 weight decay?

        # Use the built-in Adam optimizer, but might want to try gradient clipping?
        self.optimize = tf.train.AdamOptimizer(self.args.step_size_critic).minimize(self.l2_error) 

        # Then return this in the gradient step to provide to the Actor.
        # TODO should check this, it _should_ deal with gradients row-wise, and
        # then the gradient can be summed over B. Where is the summing over B?
        # Is this also equivalent if I did targ = tf.reduce_sum(self.qvals_B)? I
        # think so because it doesn't matter if we sum, action in b-th minibatch
        # only (directly) affects the b-th Q-value and has a gradient, right?
        self.action_grads = tf.gradients(self.qvals_B, self.act_t_BA)


    def _build_net(self, input_BO, acts_BO, scope):
        """ The critic network.
        
        Use ReLUs for all hidden layers. The output consists of one Q-value for
        each batch. Set `reuse=False`. I don't use batch normalization or their
        precise weight initialization.

        Unlike the critic, it uses actions here but they are NOT included in the
        first hidden layer. In addition, we do a tf.reshape to get an output of
        shape (B,), not (B,1). Seems like tf.squeeze doesn't work with `?`.
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
            return tf.reshape(qvals_B, shape=[-1])


    def update_target_net(self, smooth=True):
        """ 
        Update the target network based on the current weights. Normally we do
        this with smooth=True except for the first step, or unless we want to
        see how poorly hard updates perform generally.
        """
        if smooth:
            self.sess.run(self.update_target_smooth)
        else:
            self.sess.run(self.update_target_hard)


    def update_weights(self, f):
        """ 
        Gradient-based update of current Critic parameters.  Also return the
        action gradients for the Actor update later.
        """
        feed = {
            self.obs_t_BO:    f['obs_t_BO'],
            self.act_t_BA:    f['act_t_BA'],
            self.rew_t_B:     f['rew_t_B'],
            self.obs_tp1_BO:  f['obs_tp1_BO'],
            self.done_mask_B: f['done_mask_B']
        }
        action_grads, _ = self.sess.run([self.action_grads, self.optimize], feed)
        return action_grads
