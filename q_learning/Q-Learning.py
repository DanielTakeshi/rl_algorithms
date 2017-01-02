"""
Code for basic tabular Q-learning. This is adapted from Denny Britz's repository.

Observations for the **cliff_walking** scenario:

- I was confused why the agent always seemed to go up at the start. But then I
  found out it's because np.argmax(Q[observation]) will return the leading
  index, and at the start, it's 0 all the way, so it always picks the first one
  which is to go up (see cliff_walking.py). It's a little annoying but not a big
  deal, it works out in the end.

- The policy_fn will return one 0.925 and three 0.025s, since it will pick the
  best action and *then* adjust for epsilons. If I want to see how the Q-values
  are evolving, I need to use Q[observation] directly, not A.

- To print, use env.render(). Very useful for the grid-like settings.

- There's no extra reward for the goal state. It's just -1 for all R(s,a,s')
  unless the successor (new_position in the code) is the cliff, in which case
  it's -100.

This code is general so it doesn't have to be cliff-walking. At the end, it uses
the plotting script, but I should probably roll out a modified verison for my
own use.
"""

from __future__ import print_function
import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
if "../" not in sys.path:
    sys.path.append("../") 
from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting
matplotlib.style.use('ggplot')


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """ Creates an epsilon-greedy policy based on a given Q-function and
    epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.  Each value is a
            numpy array of length nA (see below).
        epsilon: The probability to select a random action; a float between 0
            and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns the
        probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """ Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy
    policy while following an epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Lambda time discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, episode_lengths). Q is the optimal action-value function, a
        dictionary mapping state -> action values.  stats is an EpisodeStats
        object with two numpy arrays for episode_lengths and episode_rewards.
    """
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),
                                  episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 1 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
                        
        # One step in the environment
        for t in itertools.count():

            # Take a step (note: action_probs is always a 0.925-0.025x3 split
            # due to the way the code is structured).
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            if done:
                break
            state = next_state

    print("")
    return Q, stats


if __name__ == "__main__":
    """ Testing using Denny's implementation of Cliff Walking. """
    env = CliffWalkingEnv()
    print("Now running Q-learning ...")
    Q, stats = q_learning(env, 500)
    plotting.plot_episode_stats(stats, noshow=True, figdir="figures/")
