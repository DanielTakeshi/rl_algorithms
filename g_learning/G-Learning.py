"""
(c) 2017 by Daniel Seita

G-Learning, as described in:

    Taming the Noise in Reinforcement Learning via Soft Updates (UAI 2016)
    Authors: Roy Fox*, Ari Pakman*, Naftali Tishby

This code will face the same constraints as Denny Britz's code, i.e. we need
simple tabular scenarios. I have tested G-learning on the following:

1. TODO
"""

from __future__ import print_function
import gym
import itertools
import matplotlib
import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append('../')
from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.envs.gridworld import GridworldEnv
from lib import plotting
matplotlib.style.use('ggplot')


class GLearningAgent():

    def __init__(self, env, k):
        """ For now, we'll make a limitation that we know the number of states
        and actions.  It creates:
        
        - G: Contains G(state,action) values.
        - N: Contains N(state,action) values, counts of the times each
            state-action pair was visited; needed for the alpha update.
        - V: The total discounted expected cost.
        - I: The total discounted expected information cost.
        - F: The free energy function.
        - rho: The \rho(a|s) function in the paper, the prior policy (see
            Section 3.1 in the paper). ** Assumes uniform prior!! **

        Args:
            env: An OpenAI gym environment, either custom or built-in, but be
                aware that not all of them can be used in this setting.
            k: The parameter which adjusts the beta term. Should be tested.
        """
        ns, na = env.observation_space.n, env.action_space.n
        self.G = np.zeros((ns,na))
        self.N = np.zeros((ns,na))
        self.V = np.zeros(ns) # not needed
        self.I = np.zeros(ns) # not needed
        self.F = np.zeros(ns) # not needed
        self.rho = np.ones((ns,na), dtype=float) / na
        self.k = k


    def policy_exploration(self, state, epsilon=0.0):
        """ The agent's current exploration policy. Right now we default to
        epsilon-greedy on the G(s,a) values.
        
        Args:
            state: The current state the agent is in.
            epsilon: The probability of taking a random action.
        
        Returns:
            The action to take.
        """
        num_actions = self.G.shape[1]
        action_probs = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(self.G[state,:])
        action_probs[best_action] += (1.0-epsilon)
        return np.random.choice(np.arange(num_actions), p=action_probs)
 

    def alpha_schedule(self, t, state, action):
        """ The alpha scheduling. By default, Equation 29 in the paper.
        
        Args:
            t: The iteration of the current episode.
            state: The current state.
            action: The current action.

        Returns:
            The alpha to use for the G-learning update.
        """
        alpha = self.N[state,action] ** -0.8
        assert 0 < alpha <= 1, "Error, alpha = {}".format(alpha)
        return alpha


    def beta_schedule(self, t):
        """ The beta scheduling. By default, Equation 26 in the paper.
        
        Args:
            t: The iteration of the current episode.

        Returns:
            The beta to use for the G-learning update. If it's 0, G-learning
            turns into Q^\rho-learning. If it approaches infinity, G-learning
            approaches Q-learning.
        """
        beta = self.k * t
        assert beta >= 0, "Error, beta={}, k={}, t={}".format(beta, self.k, t)
        return beta


    def g_learning(self, num_episodes, max_ep_steps=10000, discount=1.0, epsilon=0.1):
        """ The G-learning algorithm.
    
        Args:
            num_episodes: Number of episodes to run.
            max_ep_steps: Maximum time steps allocated to one episode.
            discount: Standard discount factor, usually denoted as \gamma.
            epsilon: Probability of taking random actions during exploration.
    
        Returns:
            A tuple (G, stats) of the G-values and statistics, the latter of
            which should be plotted thoroughly analyzed.
        """
        cum_t = 0
        stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes),
                                      episode_rewards=np.zeros(num_episodes))

        for i_episode in range(num_episodes):
            if (i_episode+1) % 1 == 0:
                print("\rEpisode {}/{}.".format(i_episode+1, num_episodes), end="")
                sys.stdout.flush()
            state = env.reset()

            # Run this episode until we finish as indicated by the environment.
            for t in range(1, max_ep_steps+1):
                cum_t += 1

                # Uses exploration policy to take a step, then collect
                # statistics.  The G-learning paper uses costs, not rewards.               
                action = self.policy_exploration(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                cost = -reward 
                stats.episode_rewards[i_episode] += cost
                stats.episode_lengths[i_episode] = t
                self.N[state,action] += 1

                # Intermediate terms for the G-learning update.
                alpha = self.alpha_schedule(t, state, action)
                beta = self.beta_schedule(t)
                temp = np.sum(self.rho[next_state,:] * 
                              np.exp(-beta * self.G[next_state,:]))

                # The official G-learning update at last.
                td_target = cost - (discount/beta) * np.log(temp)
                td_delta = td_target - self.G[state,action]
                self.G[state,action] += (alpha * td_delta)

                if done:
                    break
                state = next_state
    
        print("")
        return self.G, stats


if __name__ == "__main__":
    """ This will run G-learning. Be sure to double check all parameters,
    including the ones for plotting (e.g., file names).  """

    # Should test with k in {1e-3, 1e-4, 5*1e-5, 1e-6}
    env = CliffWalkingEnv()
    agent = GLearningAgent(env, k=1e-3) 
    G, stats = agent.g_learning(num_episodes=500,
                                max_ep_steps=10000,
                                discount=0.95,
                                epsilon=0.1)
    plotting.plot_episode_stats(stats,
                                smoothing_window=10,
                                noshow=False,
                                figdir="figures/cliff_",
                                dosave=True)
