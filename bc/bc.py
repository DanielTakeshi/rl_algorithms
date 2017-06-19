"""
(c) June 2017 by Daniel Seita

Behavioral cloning (continuous actions only). Tested environments:

    Hopper-v1

For results, see the README(s) nearby.

    TODO right now we assume we'll get our minibatches of data with `get_batch`
    but this is inefficient if we decide to scale up and avoid subsampling the
    data, where it would be better to have a list which supplies fixed,
    pre-computed minibatches. I should fix this later.

    TODO handle l2 regualrization? Though I have found that this doesn't have as
    good an effect as I thought it would ...
"""

import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tf_util
plt.style.use('seaborn-darkgrid')
np.set_printoptions(edgeitems=100, linewidth=100, suppress=True)


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


def load_dataset(args):
    """ Loads the dataset for BC and return training and validation splits,
    separating the observations and actions, along with observation and action
    shapes.

    This is also where we should handle the case of varying-length expert
    trajectories, even though these should be rare. (I kept those there so that
    the leading dimension is the number of trajectories, in case we want to use
    that information somehow, but right now we just mix among trajectories.) In
    addition, it might be useful to subsample the data.
    """
    # Load the numpy file and parse it.
    str_roll = str(args.num_rollouts).zfill(3)
    expert_data = np.load('expert_data/'+args.envname+'_'+str_roll+'.npy')
    expert_obs = expert_data[()]['observations']
    expert_act = expert_data[()]['actions']
    expert_ret = expert_data[()]['returns']
    expert_stp = expert_data[()]['steps']
    N = expert_obs.shape[0]
    assert N == expert_act.shape[0] == len(expert_ret) == len(expert_stp)
    obs_shape = expert_obs.shape[2]
    act_shape = expert_act.shape[2]
    print("\nobs_shape = {}\nact_shape = {}".format(obs_shape, act_shape))
    print("subsampling freq = {}".format(args.subsamp_freq))
    print("expert_steps = {}".format(expert_stp))
    print("expert_returns = {}".format(expert_ret))
    print("mean(expert_returns) = {}".format(np.mean(expert_ret))) # remember!
    print("(raw) expert_obs.shape = {}".format(expert_obs.shape))
    print("(raw) expert_act.shape = {}".format(expert_act.shape))

    # Choose a different starting point to subsample for each trajectory.
    start_indices = np.random.randint(0, args.subsamp_freq, N)
    
    # Subsample expert data, remove actions which were only for padding.
    expert_obs_l = []
    expert_act_l = []
    for i in range(N):
        expert_obs_l.append(
            expert_obs[i, start_indices[i]:expert_stp[i]:args.subsamp_freq, :]
        )
        expert_act_l.append(
            expert_act[i, start_indices[i]:expert_stp[i]:args.subsamp_freq, :]
        )

    # Concatenate everything together.
    expert_obs = np.concatenate(expert_obs_l, axis=0)
    expert_act = np.concatenate(expert_act_l, axis=0)
    print("(subsampled/reshaped) expert_obs.shape = {}".format(expert_obs.shape))
    print("(subsampled/reshaped) expert_act.shape = {}".format(expert_act.shape))
    assert expert_obs.shape[0] == expert_act.shape[0]

    # Finally, form training and validation splits.
    num_examples = expert_obs.shape[0]
    num_train = int(args.train_frac * num_examples)
    shuffled_inds = np.random.permutation(num_examples)
    train_inds, valid_inds = shuffled_inds[:num_train], shuffled_inds[num_train:]
    expert_obs_tr  = expert_obs[train_inds]
    expert_act_tr  = expert_act[train_inds]
    expert_obs_val = expert_obs[valid_inds]
    expert_act_val = expert_act[valid_inds]
    print("\n(train) expert_obs.shape = {}".format(expert_obs_tr.shape))
    print("(train) expert_act.shape = {}".format(expert_act_tr.shape))
    print("(valid) expert_obs.shape = {}".format(expert_obs_val.shape))
    print("(valid) expert_act.shape = {}\n".format(expert_act_val.shape))

    return (expert_obs_tr, expert_act_tr, expert_obs_val, expert_act_val, \
            obs_shape, act_shape)


def policy_model(data_in, action_dim):
    """ Create a neural network representing the BC policy. It will be trained
    using standard supervised learning techniques.
    
    Parameters
    ----------
    data_in: [Tensor]
        The input (a placeholder) to the network, with leading dimension
        representing the batch size.
    action_dim: [int]
        Number of actions, each of which (at least for MuJoCo) is
        continuous-valued.

    Returns
    ------- 
    out [Tensor]
        The output tensor which represents the predicted (or desired, if
        testing) action to take for the agent.
    """
    with tf.variable_scope("BCNetwork", reuse=False):
        out = data_in
        out = layers.fully_connected(out, num_outputs=100,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=100,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=action_dim,
                weights_initializer=layers.xavier_initializer(uniform=True),
                activation_fn=None)
        return out


def get_batch(expert_obs, expert_act, batch_size):
    """ 
    Obtain a minibatch of samples. Note that this is relatively inefficient, and
    if dealing with very large datasets without subsampling, use a list of
    samples instead. 
    """
    indices = np.arange(expert_obs.shape[0])
    np.random.shuffle(indices)
    xs = expert_obs[indices[:batch_size]]
    ys = expert_act[indices[:batch_size]]
    return xs, ys


def run_bc(session, args, log_dir):
    """ Runs behavioral cloning on some stored data.

    It roughly mirrors the experimental setup of [Ho & Ermon, NIPS 2016]. They
    trained using ADAM (batch size 128) on 70% of the data and trained until
    validation error on the held-out set of 30% no longer decreases. They also
    substantially subsampled their data.

    Parameters
    ----------
    session: [TF Session]
        The TensorFlow session we're using.
    args: [Arguments Namespace]
        Namedspace representing convenient arguments from the user.
    log_dir: [string]
        Where we save files to. FYI, it doesn't include the ending slash.
    """
    env = gym.make(args.envname)
    (expert_obs_tr, expert_act_tr, expert_obs_val, expert_act_val, obs_shape, \
            act_shape) = load_dataset(args)

    # Build the data and network. For now, no casting (see DQN code).
    x = tf.placeholder(tf.float32, shape=[None,obs_shape])
    y = tf.placeholder(tf.float32, shape=[None,act_shape])
    policy_fn = policy_model(data_in=x, action_dim=act_shape)

    # Construct the loss function and training information.
    l2_loss = tf.reduce_mean(
        tf.reduce_sum((policy_fn-y)*(policy_fn-y), axis=[1])
    )
    train_step = tf.train.AdamOptimizer(args.lrate).minimize(l2_loss)

    all_tr_loss = []
    all_val_loss = []
    all_iters = [] # Makes plotting easier since these are the x-coords.
    all_returns = [] # Will turn into an array of arrays later.
    session.run(tf.global_variables_initializer())

    for i in range(args.train_iters):
        b_xs, b_ys = get_batch(expert_obs_tr, expert_act_tr, args.batch_size)
        _,tr_loss = session.run([train_step, l2_loss], feed_dict={x:b_xs, y:b_ys})

        if (i % args.eval_freq == 0):
            # Only save/evaluate stuff every `args.eval_freq` iterations.
            val_loss = session.run(l2_loss, feed_dict={x:expert_obs_val, y:expert_act_val})
            returns = run_bc_test(args, session, policy_fn, x, env)
            print("iter={}   tr_loss={:.5f}   val_loss={:.5f}".format(
                str(i).zfill(4), tr_loss, val_loss))
            print("mean(returns): {}\nstd(returns): {}\n".format(
                    np.mean(returns), np.std(returns)))
            all_iters.append(i)
            all_tr_loss.append(tr_loss)
            all_val_loss.append(val_loss)
            all_returns.append(returns)

    # Store the results as numpy arrays so we can easily plot later.
    np.save(log_dir +"/iters", np.array(all_iters))
    np.save(log_dir +"/tr_loss", np.array(all_tr_loss))
    np.save(log_dir +"/val_loss", np.array(all_val_loss))
    np.save(log_dir +"/returns", np.array(all_returns))


def run_bc_test(args, session, policy_fn, x, env):
    """ Run the agent in the world! 
    
    Returns
    -------
    returns [list]
        A list of returns, one for each of the `args.test_rollouts` rollouts.
    """
    actions = []
    observations = []
    returns = []
    max_steps = env.spec.timestep_limit

    for rr in range(args.test_rollouts):
        obs = env.reset()
        done = False
        totalr = 0
        steps = 0
        while not done:
            # Take steps by expanding observation (to get shapes to match).
            exp_obs = np.expand_dims(obs, axis=0)
            action = np.squeeze(session.run(policy_fn, feed_dict={x:exp_obs}))
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render: env.render()
            if steps >= max_steps: break
        returns.append(totalr)

    return returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('num_rollouts', type=str)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--regu', type=float, default=0.0) # don't use now
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--subsamp_freq', type=int, default=20)
    parser.add_argument('--test_rollouts', type=int, default=50) # GAIL paper used 50
    parser.add_argument('--train_frac', type=float, default=0.7)
    parser.add_argument('--train_iters', type=int, default=5000)
    parser.add_argument('--render', action='store_true') # don't use now
    args = parser.parse_args()
    print("\nUsing the following arguments: {}".format(args))

    # Handle some logic with the log file and save the args there.
    log_dir = "logs/"+args.envname+"/numroll_"+args.num_rollouts+"_seed_"+str(args.seed)
    print("log_dir: {}\n".format(log_dir))
    assert not os.path.exists(log_dir), "Error: log_dir already exists!"
    os.makedirs(log_dir)
    with open(log_dir+'/args.pkl','w') as f:
        pickle.dump(args, f)

    # Create a session, handle random seeds (well, partly...) and run.
    session = get_tf_session()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    run_bc(session, args, log_dir)
