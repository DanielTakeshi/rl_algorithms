# Standard Tabular Q-learning

## Cliff World

### Environment

I tested with `CliffWorldEnv` using the following settings:

- number of episodes = 500
- discount factor = 0.95
- alpha = 0.5 for the temporal difference update
- epsilon = 0.1 (for greedy-epsilon exploration, the actions themselves are deterministic once we have one)

For rewards, the agent gets a -1 living reward, gets 0 if it manages to get to the bottom right corner, but gets -100 if it falls off the cliff. The agent starts at the bottom left corner and can move in one of four directions deterministically (but the **exploration policy** is epsilon-greedy).

### Results

First, episode reward over time, smoothed over a window of 10:

![Rewards of episodes](/figures/cliff_episode_reward_time.png)

Early on we get some -100s due to falling off the cliff, but later we get the theoretical best possible of -12. However, because Q-learning is off policy, it learns the path that goes directly next to the cliff, so in **exploration** it will often fall off the cliff due to greedy-epsilon. That's the reason for the curvy nature of the graph.

Next, we have the length of episodes (i.e. number of time steps). The minium we can get is 1 since the agent could actually dive into the pit on the first action. If the agent is following the "risky" path during training and manages to make it to the end (despite the randomness in the epsilon-greedy policy) then that's 13 steps.

![Length of episodes](/figures/cliff_episode_length_time.png)

For the most part, this is what I'd expect. We want episodes to be shorter later because the agent "knows" where it's going now.
