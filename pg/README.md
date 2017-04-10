# Policy Gradients

This repository started out as the homework assignment from CS 294-112, Deep
Reinforcement Learning, at UC Berkeley. See `cs294-112_stuff/homework.md` for
details. Here's the first figure they wanted:

![part01](figures/part_01.png?raw=true)

I think it looks good. Their "-300" requirement for the reward is a bit unclear.
The performance seems to reach that level but sometimes dips below. I ran this
with the default settings, except for increasing the number of iterations from
300 to 500.

Here's the next part. See **Cartpole**, with three seeds run:

![cartpole](figures/cartpole_comparison.png?raw=true)

And then the smoothed version, where I took averages over the three trials:

![cartpole_sm](figures/cartpole_comparison_sm.png?raw=true)

Here's the next part. See **Cartpole**, with three seeds run:

![pendulum](figures/pendulum_comparison.png?raw=true)

And then the smoothed version:

![pendulum_sm](figures/pendulum_comparison_sm.png?raw=true)

Hmmm ... for these problems, there isn't much advantage to using a neural
network value function approximator, because these problems are too simple.

For hyperparameters, look at the `main.py` script. I may decide to write them
more formally later. But at this point, I'd really like to move on to writing
something more advanced such as Trust Region Policy Optimization.
