# Standard Tabular G-learning (not Q-Learning!)

This is mainly to benchmark against tabular Q-learning, [which I've implemented here](https://github.com/DanielTakeshi/rl_algorithms/tree/master/q_learning).

**Note**: The paper reference for G-learning [1] uses *costs*, not rewards, so whenever I say reward here, just negate it and view it as the cost. Thus, the reward of -100 for Q-learning is really a cost of 100 for G-learning.

## Cliff World

### Environment

I tested with `CliffWorldEnv` using the following settings:

- number of episodes = 1000
- discount factor = 0.95
- alpha = n(s,a)^(-0.8) for the temporal difference update, where n(s,a) is the number of times that (s,a) have been visited. It's Equation 29 in reference [1].
- epsilon = 0.1 for greedy-epsilon exploration. Once the agent executes an action, it's deterministic.
- k = 1e-3, a special scheduling parameter for the \beta term in G-learning, which not used in Q-learning. I also tuned with 1e4, 5\*1e-5, and 1e-6 as reported in [1] and got similar results.

For rewards, the agent gets a -1 living reward, gets 0 if it manages to get to the bottom right corner, but gets -100 if it falls off the cliff. The agent starts at the bottom left corner and can move in one of four directions deterministically (but the **exploration policy** is epsilon-greedy based on the G(s,a) values).

**Note**: In [1], the reported cost is 5 for going off the cliff, but I made it 100 because with the discount factor we have here, I don't see how the agent can learn to go to the goal. It takes a cost of 5 to immediately jump off the cliff, but it would take about a cost of 10 to go to the goal ... so the agent would prefer to jump off the cliff in the first move? When I ran Q-learning with this, Q-learning indeed learned to jump off the cliff right away.

### Results

First, episode **COSTS** over the trials, smoothed over a window of five trials. I know it says "rewards" in the figure but it's really "costs". We want the curve to decrease.

![Rewards of episodes](figures/cliff_episode_reward_time.png?raw=true)

Unfortunately, the agent doesn't seem to be learning anything except to jump off the cliff right away and incur a cost of 100. This is what I see when rendering the environment on the command line. 

We never see costs below 100, which would happen if the agent took a path (either the riksy or the safe one) to go to the correct goal state.

Next, we have the length of episodes (i.e. number of time steps). The minium we can get is 1 since the agent could actually dive into the pit on the first action. If the agent is following the "risky" path during training and manages to make it to the end (despite the randomness in the epsilon-greedy policy) then that's 13 steps. 

![Length of episodes](figures/cliff_episode_length_time.png?raw=true)

Unfortunately this isn't what I want to see. I actually had to cap the number of steps in an episode (which I didn't need to do for G-learning).

I'm not sure what's going on.

### References

[1] Taming the Noise in Reinforcement Learning via Soft Updates, UAI 2016
