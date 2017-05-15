"""
This is Natural Evolution Strategies, designed to run on one computer and not a
cluster.

(c) May 2017 by Daniel Seita, though obviously based on OpenAI's work/idea.
"""

import gym
import logz
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
import utils
from collections import defaultdict
from gym import wrappers
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
        tf.set_random_seed(args.seed)
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


    def _compute_return(self, test=False, store_info=False):
        """ Runs the current neural network policy. 

        For now, we assume we run one episode. Also, we expand the observations
        to get a dummy dimension, in case we figure out how to make use of
        minibatches later.
        
        Args:
            test True if testing, False if part of training. The testing could
                be either the tests done after each weight update, or the tests
                done as a result fo the `test` method.
            store_info: True if storing info is desired, meaning that we return
                observations and actions.

        Returns:
            The scalar return to be evaluated by the ES agent.
        """
        max_steps = self.env.spec.timestep_limit
        obs = self.env.reset()
        done = False
        steps = 0
        total_rew = 0
        observations = []
        actions = []

        while not done:
            exp_obs = np.expand_dims(obs, axis=0)
            action = self.sess.run(self.sampled_ac, {self.ob_no:exp_obs})
            observations.append(obs)
            actions.append(action)
            
            # Apply the action *after* storing the current obs/action pair.
            obs, r, done, _ = self.env.step(action)
            total_rew += r
            steps += 1
            if self.args.render and test:
                self.env.render()
            if steps >= max_steps or done:
                break

        if store_info:
            return total_rew, observations, actions
        else:
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
            - Antithetic (i.e. mirrored) sampling.
            - Rank transformation, using OpenAI's code.

        Tricks avoided:
            - Fixed Gaussian block. I like to just regenerate here.
            - Virtual batch normalization, seems to be only for Atari games.
            - Weight decay. Not sure how to do this.
            - Action discretization. For now, it adds extra complexity.

        Final weights are saved and can be pre-loaded elsewhere.
        """
        args = self.args
        t_start = time.time()

        for i in range(args.es_iters):
            if (i % args.log_every_t_iter == 0):
                print("\n************ Iteration %i ************"%i)
            stats = defaultdict(list)

            # Set stuff up for perturbing weights and determining fitness.
            weights_old = self.sess.run(self.weights_v) # Shape (numw,)
            eps_nw = np.random.randn(args.npop, self.num_ws)
            scores_n2 = []

            for j in range(args.npop):
                # Mirrored sampling, positive case, +eps_j.
                weights_new_pos = weights_old + args.sigma * eps_nw[j]
                self.sess.run(self.set_params_op, 
                              feed_dict={self.new_weights_v: weights_new_pos})
                rews_pos = self._compute_return()

                # Mirrored sampling, negative case, -eps_j.
                weights_new_neg = weights_old - args.sigma * eps_nw[j]
                self.sess.run(self.set_params_op, 
                              feed_dict={self.new_weights_v: weights_new_neg})
                rews_neg = self._compute_return()

                scores_n2.append([rews_pos,rews_neg])

            # Determine the new weights based on OpenAI's rank updating.
            proc_returns_n2 = utils.compute_centered_ranks(np.array(scores_n2))
            F_n = proc_returns_n2[:,0] - proc_returns_n2[:,1]
            grad = np.dot(eps_nw.T, F_n)

            # Apply the gradient update. TODO: Change this to ADAM.
            alpha = (args.lrate_es / (args.sigma*args.npop*2))
            next_weights = weights_old + alpha * grad
            self.sess.run(self.set_params_op, 
                          feed_dict={self.new_weights_v: next_weights})

            # Test roll-outs with these new weights.
            returns = []
            for _ in range(args.test_trajs):
                returns.append(self._compute_return(test=True))
            
            # Report relevant logs.
            if (i % args.log_every_t_iter == 0):
                minutes = (time.time()-t_start) / 60.
                logz.log_tabular("ScoresAvg",        np.mean(scores_n2))
                logz.log_tabular("ScoresStd",        np.std(scores_n2))
                logz.log_tabular("ScoresMax",        np.max(scores_n2))
                logz.log_tabular("ScoresMin",        np.min(scores_n2))
                logz.log_tabular("FinalAvgReturns",  np.mean(returns))
                logz.log_tabular("FinalStdReturns",  np.std(returns))
                logz.log_tabular("FinalMaxReturns",  np.max(returns))
                logz.log_tabular("FinalMinReturns",  np.min(returns))
                logz.log_tabular("TotalTimeMinutes", minutes)
                logz.log_tabular("TotalIterations",  i)
                logz.dump_tabular()

            # Save the weights so I can test them later.
            if (i % args.snapshot_every_t_iter == 0):
                itr = str(i).zfill(len(str(abs(args.es_iters))))
                with open(self.log_dir+'/snapshots/weights_'+itr+'.pkl', 'wb') as f:
                    pickle.dump(next_weights, f)

        # Save the *final* weights.
        itr = str(i).zfill(len(str(abs(args.es_iters))))
        with open(self.log_dir+'/snapshots/weights_'+itr+'.pkl', 'wb') as f:
            pickle.dump(next_weights, f)


    def test(self, just_one=True):
        """ This is for test-time evaluation. No training is done here. By
        default, iterate through every snapshot.  If `just_one` is true, this
        only runs one set of weights, to ensure that we record right away since
        OpenAI will only record subsets and less frequently.  Changing the loop
        over snapshots is also needed.
        """
        os.makedirs(self.args.directory+'/videos')
        self.env = wrappers.Monitor(self.env, self.args.directory+'/videos', force=True)

        headdir = self.args.directory+'/snapshots/'
        snapshots = os.listdir(headdir)
        snapshots.sort()
        num_rollouts = 10
        if just_one:
            num_rollouts = 1

        for sn in snapshots:
            print("\n***** Currently on snapshot {} *****".format(sn))

            ### Add your own criteria here.
            # if "800" not in sn:
            #     continue
            ###

            with open(headdir+sn, 'rb') as f:
                weights = pickle.load(f)
            self.sess.run(self.set_params_op, 
                          feed_dict={self.new_weights_v: weights})
            returns = []
            for i in range(num_rollouts):
                returns.append( self._compute_return(test=True) )
            print("mean: \t{}".format(np.mean(returns)))
            print("std: \t{}".format(np.std(returns)))
            print("max: \t{}".format(np.max(returns)))
            print("min: \t{}".format(np.min(returns)))
            print("returns:\n{}".format(returns))


    def generate_rollout_data(self, weights, num_rollouts=100,
            trajs_not_transits=False):
        """ Roll out the expert data and save the observations and actions for
        imitation learning later.

        The output will depend on `trajs_not_transits`. If False, then
        observations/actions are stored in continuous lists using Python's
        `extend` keyword. if True, then we save them separately and have an
        extra leading dimension. For instance, with InvertedPendulum saving at
        the transit level with 100 rollouts, the observations and actions might
        have shapes (100000,4) and (100000,1). With trajectories, the shapes
        might be (100,1000,4) and (100,1000,1). This assumes that each of the
        100 rollouts of IP-v1 gets the perfect 1000 score, which seems to
        coincide with the number of timesteps.

        By the way, the expert roll-outs may not have the same shape. Use the
        `ENV_TO_OBS_SHAPE` to guard against this scenario. We zero-pad if
        needed.

        TL;DR: leading dimension is the minibatch, second leading dimension is
        the timestep, third is the obs/act shape. We *may* need a fourth but if
        so let's just linearize so that we don't have to worry about it.

        Args:
            weights: The desired weight vector.
            num_rollouts: The number of expert rollouts to save.
            trajs_not_transits: If True, save at the level of *trajectories*.
        """
        # These are the shapes we need **for each trajectory**.
        ENV_TO_OBS_SHAPE = {"InvertedPendulum-v1": (1000,4)}
        ENV_TO_ACT_SHAPE = {"InvertedPendulum-v1": (1000,1)}
        if self.args.envname not in ENV_TO_OBS_SHAPE:
            print("Error, this environment is not supported.")
            sys.exit()
    
        headdir = self.args.directory+ '/expert_data'
        if not os.path.exists(headdir):
            os.makedirs(headdir)
        self.sess.run(self.set_params_op, feed_dict={self.new_weights_v: weights})
        returns = []
        observations = []
        actions = []

        for i in range(num_rollouts):
            print("rollout {}".format(i))
            rew, obs_l, acts_l = self._compute_return(test=False, store_info=True)
            returns.append(rew)

            # Save at the *trajectory* or *transit* level.
            if trajs_not_transits:
                observations.append(obs_l)
                actions.append(acts_l)
            else:
                observations.extend(obs_l)
                actions.extend(acts_l)

        print("returns", returns)
        print("mean return", np.mean(returns))
        print("std of return", np.std(returns))

        # Fix padding issue to make lists have the same shape; we later make an
        # array.  Check each (ol,al), tuple of lists, to ensure shapes match. If
        # the obs-list doesn't match, neither will the act-list, so test one.
        if trajs_not_transits:
            for (i,(ol,al)) in enumerate(zip(observations,actions)):
                obs_l = np.array(ol)
                act_l = np.array(al)
                print("{} {} {}".format(i, obs_l.shape, act_l.shape))
                if obs_l.shape != ENV_TO_OBS_SHAPE[self.args.envname]:
                    result_o = np.zeros(ENV_TO_OBS_SHAPE[self.args.envname])
                    result_a = np.zeros(ENV_TO_ACT_SHAPE[self.args.envname])
                    result_o[:obs_l.shape[0],:obs_l.shape[1]] = obs_l
                    result_a[:act_l.shape[0],:act_l.shape[1]] = act_l
                    print("revised shapes: {} {}".format(result_o.shape, result_a.shape))
                    obs_l = result_o
                    act_l = result_a
                observations[i] = obs_l
                actions[i] = act_l

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        # Save the data
        print("obs-shape = {}".format(expert_data['observations'].shape))
        print("act-shape = {}".format(expert_data['actions'].shape))
        str_roll = str(num_rollouts).zfill(4)
        name = headdir+ "/" +self.args.envname+ "_" +str_roll+ "rollouts_trajs" \
                +str(trajs_not_transits)
        np.save(name, expert_data)
        print("Expert data has been saved in: {}.npy".format(name))
