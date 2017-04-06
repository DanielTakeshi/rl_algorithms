"""
(c) April 2017 by Daniel Seita

Behavioral cloning. Tested environments:

    Hopper-v1

Some environments may have state or action dimensions which may cause the code
to choke, so I should keep an eye out for those.

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


def get_batch(expert_obs, expert_act, batch_size):
    """ Obtain a minibatch of samples. """
    indices = np.arange(expert_obs.shape[0])
    np.random.shuffle(indices)
    xs = expert_obs[indices[:batch_size]]
    ys = expert_act[indices[:batch_size]]
    return xs, ys


def run_bc(session, args):
    """ Runs behavioral cloning on some stored data.

    It tries to mirror Ho & Ermon 2016. They trained on 70% of the data and
    trained until validation error on the held-out set of 30% no longer
    decreases. That doesn't seem to work well for us so I'll just eyeball the
    validation (it's still important for tuning parameters). They trained with
    ADAM and with minibatch sizes of 128.

    Args:
        session: A Tensorflow session.
        args: The argparse from the user.
    """

    # Load the expert rollout data.
    str_roll = str(args.num_rollouts).zfill(4)
    expert_data = np.load('data/'+args.envname+'_'+str_roll+'.npy')
    expert_obs = expert_data[()]['observations']
    expert_act = np.squeeze(expert_data[()]['actions'])
    N = expert_obs.shape[0]
    assert N == expert_act.shape[0]
    obs_shape = list(expert_obs.shape)[1:]
    act_shape = list(expert_act.shape)[1:]

    # Form training and validation splits.
    num_tr = int(0.7*N)
    indices = np.arange(N)
    np.random.shuffle(indices)
    expert_obs_tr  = expert_obs[indices[:num_tr]]
    expert_act_tr  = expert_act[indices[:num_tr]]
    expert_obs_val = expert_obs[indices[num_tr:]]
    expert_act_val = expert_act[indices[num_tr:]]
    print("\n(tr) expert_obs.shape = {}".format(expert_obs_tr.shape))
    print("(tr) expert_act.shape = {}".format(expert_act_tr.shape))
    print("(val) expert_obs.shape = {}".format(expert_obs_val.shape))
    print("(val) expert_act.shape = {}\n".format(expert_act_val.shape))

    # Build the data and network. For now, no casting (see DQN code).
    x = tf.placeholder(tf.float32, shape=[None]+obs_shape)
    y = tf.placeholder(tf.float32, shape=[None]+act_shape)
    policy_fn = policy_model(data_in=x, 
                             action_dim=expert_act.shape[1], 
                             regu=args.regu,
                             scope='policy')

    # Construct the loss function and training information.
    reg_l2_loss = tf.reduce_mean(
        tf.reduce_sum((policy_fn-y)*(policy_fn-y), axis=[1])
    )
    train_step = tf.train.AdamOptimizer(args.lrate).minimize(reg_l2_loss)

    # Train the network using minibatches.
    session.run(tf.global_variables_initializer())
    for i in range(args.train_iters):
        b_xs, b_ys = get_batch(expert_obs_tr, expert_act_tr, args.batch_size)
        _,tr_loss = session.run([train_step, reg_l2_loss], 
                                feed_dict={x:b_xs, y:b_ys})
        val_loss = session.run(reg_l2_loss, 
                               feed_dict={x:expert_obs_val, y:expert_act_val})
        if (i % 50 == 0):
            print("iter={}   tr_loss={:.5f}   val_loss={:.5f}".format(
                str(i).zfill(4), tr_loss, val_loss))

    # Now run the agent in the world!
    print("\nRunning the agent with our behaviorally cloned net.")
    env = gym.make(args.envname)
    actions = []
    observations = []
    returns = []
    max_steps = env.spec.timestep_limit

    for rr in range(args.test_iters):
        print("rollout {}".format(rr))
        obs = env.reset()
        done = False
        totalr = 0
        steps = 0
        while not done:
            # Take steps by expanding observation (to get shapes to match).
            exp_obs = np.expand_dims(obs, axis=0)
            action = np.squeeze( session.run(policy_fn, feed_dict={x:exp_obs}) )
            observations.append(obs)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if (steps % 100 == 0): 
                print("  {}/{} with totalr={}".format(steps, max_steps, totalr))
            if steps >= max_steps:
                break
            if done:
                print("done at step {} with totalr={}".format(steps, totalr))
        returns.append(totalr)

    # Store the results so we can plot later.
    print("\nreturns:\n{}".format(returns))
    print("\nmean={:.4f}   std={:.4f}".format(np.mean(returns), np.std(returns)))
    results = {'returns':returns, 'mean':np.mean(returns), 'std':np.std(returns)}
    s1 = str(args.num_rollouts).zfill(4)
    s2 = str(args.train_iters).zfill(5)
    s3 = str(args.batch_size).zfill(3)
    s4 = str(args.lrate)
    s5 = str(args.regu)
    np.save('results/'+args.envname+'_'+s1+'_'+s2+'_'+s3+'_'+s4+'_'+s5, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('num_rollouts', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--regu', type=float, default=0.001)
    parser.add_argument('--train_iters', type=int, default=5000)
    parser.add_argument('--test_iters', type=int, default=50)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    session = get_tf_session()
    run_bc(session, args)
    print("All done!")
