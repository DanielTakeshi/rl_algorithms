"""
(c) April 2017 by Daniel Seita

Behavioral cloning. Tested environments:

    Hopper-v1

For results, see the README.
"""

import argparse
import gym
import numpy as np
import pickle
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tf_util


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


def policy_model(data_in, action_dim, scope, reuse=False):
    """ Creating a neural network.
    TODO double check all this ...
    
    Args:
        TODO
    """
    with tf.variable_scope(scope, reuse=reuse):
        out = data_in
        out = layers.flatten(out)
        out = layers.fully_connected(out, num_outputs=100, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=100, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=action_dim, activation_fn=None)
        return out


def run_bc(session, args):
    """ Runs behavioral cloning on some stored data.

    Args:
        TODO
    """
    expert_data = np.load('data/'+args.envname+'.npy')
    expert_obs = expert_data[()]['observations']
    expert_act = expert_data[()]['actions']
    N = expert_obs.shape[0]
    assert N == expert_act.shape[0]
    obs_shape = list(expert_obs.shape)[1:]
    act_shape = list(expert_act.shape)[1:]

    # Build the data and network. For now, no casting (see DQN code).
    x  = tf.placeholder(tf.float32, shape=[None]+obs_shape)
    y_ = tf.placeholder(tf.float32, shape=[None]+act_shape)
    pi = policy_model(data_in=x, action_dim=expert_act.shape[1], scope='policy')

    # Construct the loss function and training information.
    reg_error = tf.add_n([sum(tf.square(p)) for p in params])
    loss = tf.nn.l2_loss( pi - y_ ) / N + reg_error
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Train the network using minibatches. TODO print the l2 loss each
    # iteration, it's the only way we can keep track of it.
    for i in range(1000):
        batch_xs, batch_ys = ... # todo implement this function
        session.run(train_fn, feed_dict={x=batch_xs, y_=batch_ys})
        if (i % 10 == 0):
            # Maybe have a separate session running here and which will return
            # the lsos function? As in session.run([train_fn, loss],
            # feed_dict={...}) ?
            print(session.run(...))


    # Debugging ...
    print(x.get_shape())
    print(y_.get_shape())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    args = parser.parse_args()
    session = get_tf_session()
    run_bc(session, args)
    print("All done!")
