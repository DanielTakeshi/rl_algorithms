# Standard Tabular Q-learning

## Cliff World

### Environment

I tested with `CliffWorldEnv` using the following settings:

- number of episodes = 1000
- discount factor = 0.95
- alpha = n(s,a)^(-0.8) for the temporal difference update, where n(s,a) is the number of times that (s,a) have been visited. It's Equation 29 in reference [1].
- epsilon = 0.1 for greedy-epsilon exploration. Once the agent executes an action, it's deterministic.

For rewards, the agent gets a -1 living reward, gets 0 if it manages to get to the bottom right corner, but gets -100 if it falls off the cliff. The agent starts at the bottom left corner and can move in one of four directions deterministically (but the **exploration policy** is epsilon-greedy based on the Q(s,a) values).

### Results

First, episode reward over the trials, smoothed over a window of five trials:

![Rewards of episodes](figures/cliff_episode_reward_time.png?raw=true)

Early on we get some -100s due to falling off the cliff, but later we get closer the theoretical best possible of -12. However, because Q-learning is off policy, it learns the path that goes directly next to the cliff, so in **exploration** it will often fall off the cliff due to greedy-epsilon. That's the reason for the spiky nature of the graph. When I printed the environment in the last round of training, the agent would always take the risky path, sometimes succeeding, sometimes failing.

Next, we have the length of episodes (i.e. number of time steps). The minium we can get is 1 since the agent could actually dive into the pit on the first action. If the agent is following the "risky" path during training and manages to make it to the end (despite the randomness in the epsilon-greedy policy) then that's 13 steps.

![Length of episodes](figures/cliff_episode_length_time.png?raw=true)

For the most part, this is what I'd expect. We want episodes to be shorter later because the agent "knows" where it's going now.

### References

[1] Taming the Noise in Reinforcement Learning via Soft Updates, UAI 2016
