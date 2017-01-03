"""
Code for basic tabular Q-learning. This is adapted from Denny Britz's
repository. I updated it to use a class to more closely match the G-learning
code.

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

Other observations:

- This code is general so it doesn't have to be cliff-walking, **but** you have
  to be careful to use an environment that gives a single number as a state, not
  a continuous state (e.g., unfortunately CartPole wouldn't work here). I can
  use FrozenLake-v0 but even a 4x4 space requires 10k or so iterations to see
  improvement.

- At the end, it uses the plotting script, but I should probably roll out a
  modified verison for my own use.
"""

from __future__ import print_function
import gym
import itertools
import matplotlib
import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../") 
from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.envs.gridworld import GridworldEnv
from lib import plotting
matplotlib.style.use('ggplot')


class QLearningAgent():

    def __init__(self, env):
        """ For now, we'll make a limitation that we know the number of states
        and actions.  It creates:
        
        - Q: Contains Q(state,action) values.
        - N: Contains N(state,action) values, counts of the times each
            state-action pair was visited; needed for some alpha updates.

        Args:
            env: An OpenAI gym environment, either custom or built-in, but be
                aware that not all of them can be used in this setting.
        """
        ns, na = env.observation_space.n, env.action_space.n
        self.Q = np.zeros((ns,na))
        self.N = np.zeros((ns,na))


    def policy_exploration(self, state, epsilon=0.0):
        """ The agent's current exploration policy. Right now we default to
        epsilon-greedy on the Q(s,a) values.
        
        Args:
            state: The current state the agent is in.
            epsilon: The probability of taking a random action.
        
        Returns:
            The action to take.
        """
        na = self.Q.shape[1]
        action_probs = np.ones(na, dtype=float) * epsilon / na
        best_action = np.argmax(self.Q[state,:])
        action_probs[best_action] += (1.0-epsilon)
        return np.random.choice(np.arange(na), p=action_probs)
 

    def alpha_schedule(self, t, state, action):
        """ The alpha scheduling.
        
        Args:
            t: The iteration of the current episode (t >= 1).
            state: The current state.
            action: The current action.

        Returns:
            The alpha to use for the Q-learning update.
        """
        # return 0.5     # the most basic strategy
        alpha = self.N[state,action] ** -0.8
        assert 0 < alpha <= 1, "Error, alpha = {}".format(alpha)
        return alpha


    def q_learning(self, num_episodes, max_ep_steps=10000, discount=1.0, epsilon=0.1):
        """ The Q-learning algorithm.
    
        Args:
            num_episodes: Number of episodes to run.
            max_ep_steps: Maximum time steps allocated to one episode.
            discount: Standard discount factor, usually denoted as \gamma.
            epsilon: Probability of taking random actions during exploration.
    
        Returns:
            A tuple (Q, stats) of the Q-values and statistics, which should be
            plotted and thoroughly analyzed.
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

                # Uses exploration policy to take a step.
                action = self.policy_exploration(state, epsilon)
                next_state, reward, done, _ = env.step(action)

                # Collect statistics (cum_t currently not used).
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t
                self.N[state,action] += 1
                cum_t += 1

                # The official Q-learning update.
                alpha = self.alpha_schedule(t, state, action)
                best_next_action = np.argmax(self.Q[next_state,:])
                td_target = reward + discount * self.Q[next_state,best_next_action]
                td_delta = td_target - self.Q[state,action]
                self.Q[state,action] += (alpha * td_delta)

                if done:
                    break
                state = next_state
    
        print("")
        return self.Q, stats


if __name__ == "__main__":
    """ This will run Q-learning. Be sure to double check all parameters,
    including the ones for plotting (e.g., file names).  """

    env = CliffWalkingEnv()
    agent = QLearningAgent(env) 
    Q, stats = agent.q_learning(num_episodes=1000,
                                max_ep_steps=500,
                                discount=0.95,
                                epsilon=0.1)
    plotting.plot_episode_stats(stats,
                                smoothing_window=5,
                                noshow=False,
                                figdir="figures/cliff_",
                                dosave=True)
