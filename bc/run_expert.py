"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:

    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)

(Daniel) I save an array of trajectories of shape 

    (numtrajs, numtimes, obs_dim)  // observations
    (numtrajs, numtimes, act_dim)  // actions, squeezed as needed
    // and also a list of returns and steps, each of length `numtrajs`.

However this requires padding some zeros at the end for trajectories that didn't
manage to finish (should be rare with experts, but it can still happen). Thus, I
also save a trajectory *lengths* array which can tell us when to stop dealing
with a trajectory.
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        all_observations = []
        all_actions = []
        all_steps = []
        all_returns = []

        for i in range(args.num_rollouts):
            print('roll/traj', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            observations = []
            actions = []
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            all_returns.append(totalr)
            all_steps.append(steps)

            # Ensure that observations and actions lengths are at max_steps.
            # To make it easy, just append the last obs/action since they are
            # automatically the correct dimension, reduces headaches.
            while steps < max_steps:
                observations.append(obs)
                actions.append(action)
                steps += 1
            assert len(observations) == max_steps, "{}".format(len(observations))
            assert len(actions) == max_steps, "{}".format(len(actions))
            all_observations.append(observations)
            all_actions.append(actions)

        # Squeezing since we know MuJoCo does some (1,D)-dim actions.
        expert_data = {'observations': np.array(all_observations),
                       'actions': np.squeeze(np.array(all_actions)),
                       'returns': all_returns,
                       'steps': all_steps}

        print('steps', all_steps)
        print('returns', all_returns)
        print('mean return', np.mean(all_returns))
        print('std of return', np.std(all_returns))
        print("obs.shape = {}".format(expert_data['observations'].shape))
        print("act.shape = {}".format(expert_data['actions'].shape))

        if args.save:
            str_roll = str(args.num_rollouts).zfill(3)
            np.save("expert_data/" +args.envname+ "_" +str_roll, expert_data)
            print("expert data has been saved.")


if __name__ == '__main__':
    main()
