"""
(c) April 2017 by Daniel Seita

Behavioral cloning. Tested environments:

    Hopper-v1

For results, see the README.
"""

import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tf_util

plt.style.use('seaborn-darkgrid')
np.set_printoptions(edgeitems=100,
                    linewidth=100,
                    suppress=True)


def get_tf_session():
    """ Returning a session. Set options here if desired. """
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


def policy_model(data_in, action_dim, regu, scope, reuse=False):
    """ Creating a neural network.
    
    Args:
        data_in: A Tensorflow placeholder for the input.
        action_dim: The action dimension.
        regu: The regularization constant.
        scope: For naming variables (not useful to us now).
        reuse: Whether to reuse the weights or not (ignore it).
    """
    with tf.variable_scope(scope, reuse=reuse):
        out = data_in
        out = layers.fully_connected(out,
                                     num_outputs=100,
                                     weights_regularizer=layers.l2_regularizer(regu),
                                     activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out,
                                     num_outputs=100,
                                     weights_regularizer=layers.l2_regularizer(regu),
                                     activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out,
                                     num_outputs=action_dim,
                                     weights_regularizer=layers.l2_regularizer(regu),
                                     activation_fn=None)
        return out


def get_batch(expert_obs, expert_act, batch_size, N):
    """ Obtain a minibatch of samples. """
    indices = np.arange(N)
    np.random.shuffle(indices)
    xs = expert_obs[indices[:batch_size]]
    ys = expert_act[indices[:batch_size]]
    return xs, ys


def run_bc(session, args):
    """ Runs behavioral cloning on some stored data.

    Args:
        session: A Tensorflow session.
        args: The argparse from the user.
    """
    str_roll = str(args.num_rollouts).zfill(4)
    expert_data = np.load('data/'+args.envname+'_'+str_roll+'.npy')
    expert_obs = expert_data[()]['observations']
    expert_act = np.squeeze(expert_data[()]['actions'])
    N = expert_obs.shape[0]
    assert N == expert_act.shape[0]
    obs_shape = list(expert_obs.shape)[1:]
    act_shape = list(expert_act.shape)[1:]
    print("expert_obs.shape = {}".format(expert_obs.shape))
    print("expert_act.shape = {}".format(expert_act.shape))

    # Build the data and network. For now, no casting (see DQN code).
    x = tf.placeholder(tf.float32, shape=[None]+obs_shape)
    y = tf.placeholder(tf.float32, shape=[None]+act_shape)
    policy_fn = policy_model(data_in=x, 
                             action_dim=expert_act.shape[1], 
                             regu=args.regu,
                             scope='policy')

    # Construct the loss function and training information.
    weights = tf.trainable_variables()
    reg_l2_loss = tf.nn.l2_loss(policy_fn-y)/args.batchsize
    train_step = tf.train.AdamOptimizer(args.lrate).minimize(reg_l2_loss)

    # Train the network using minibatches.
    session.run(tf.global_variables_initializer())
    for i in range(args.num_train_iter):
        batch_xs, batch_ys = get_batch(expert_obs, expert_act, args.batchsize, N)
        _,loss = session.run([train_step, reg_l2_loss], 
                             feed_dict={x:batch_xs, y:batch_ys})
        if (i % 100 == 0):
            print("iter={}, loss={}".format(str(i).zfill(3),loss))

    # Now run the agent in the world!
    print("\nRunning the agent with our behaviorally cloned net.")
    env = gym.make(args.envname)
    actions = []
    observations = []
    returns = []
    max_steps = env.spec.timestep_limit

    for rr in range(5):
        print("rollout {}".format(rr))
        obs = env.reset()
        done = False
        totalr = 0
        steps = 0

        while not done:
            # I think this is how we take steps.
            exp_obs = np.expand_dims(obs, axis=0)
            action = np.squeeze( session.run(policy_fn, feed_dict={x:exp_obs}) )
            observations.append(obs)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            env.render()
            if (steps % 100 == 0): 
                print("  {}/{} with totalr={}".format(steps, max_steps, totalr))
            if steps >= max_steps:
                break
            if done:
                print("done at step {} with totalr={}".format(steps, totalr))
        returns.append(totalr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('num_rollouts', type=str)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--regu', type=float, default=0.001)
    parser.add_argument('--num_train_iter', type=int, default=1000)
    args = parser.parse_args()
    session = get_tf_session()
    run_bc(session, args)
    print("All done!")
