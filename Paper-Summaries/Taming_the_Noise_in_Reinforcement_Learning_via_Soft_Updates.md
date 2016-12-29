# Taming the Noise in Reinforcement Learning via Soft Updates

This paper introduces a new algorithm, *G-Learning*, which is similar to Q-Learning except it tries to avoid the issue of bias when taking the max over actions. This was a problem observed in the Double Deep-Q-Network paper. It does so by using regularization and soft updates. In particular, it adds a penalizing term to the cost-to-go for divergence (uses information theory) from a prior policy.


## Introductory Stuff

The formulation is the usual MDP with costs, not rewards, so I should be able to grasp it. The most annoying part is the use of \theta as a *function*, which provides the costs. Another trick with Equation 4 is to generalize the standard Q-Learning update by using their \pi function. Normally it's 1 for the best action but it doesn't *have* to be that way. Remember: there's two things here, the *exploration* policy will drive our agent through the world and generate samples, which are then used in the Q-Learning *update rule* (usually max_a, here min_a probably).

Simple example: E[min_a Q(s,a)] <= min_a E[Q(s,a)] = min_a Q\*(s,a), using Jensen's inequality, where Q\* is the optimal action-value.

I don't understand this justification (Section 2.3):

> We conclude that the real value $V^\pi$ of the greedy policy (5) is suboptimal only in the intermediate regime, when the gap is of the order of the noise, and neither is small.

Their algorithm is designed to reach the same end goal as standard Q-Learning, but perhaps does so faster by improving bias during the learning process. The "bias" here refers to the estimation of the Q(s,a) values (sorry, hope that's clear?).


## The Algorithm: G-Learning

Lots of notation:

- \rho is our prior policy
- \pi is our learned policy
- g^\pi is the information cost of \pi
- I^\pi is the total expected, discounted information cost
- F^\pi = V^\pi + (1/beta)I^\pi is the free energy
- G^\pi is the state-action free energy function, analogous to the Q-value! I see.

TODO


## Experiments

TODO


## My Thoughts and Takeaways

This might be more straightforward for me to immediately get to work on, since it's a variation of the very-familiar Q-Learning algorithm. In addition, this description of G-Learning in the paper makes it sound not too different from Q-Learning:

> With only a small sample to go by, G-learning prefers a more randomized policy, and as samples accumulate, it gradually shifts to a more deterministic and exploiting policy.

I'm concerned about the need for a prior policy, and also if adding a penalized term is enough to make this algorithm better than Q-Learning. To add more to the confusion, G-Learning is supposed to take a "Frequentist" view, and not be like "Bayesian Q-Learning." TODO
