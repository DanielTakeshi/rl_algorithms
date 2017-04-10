# Policy Gradients

This repository started out as the homework assignment from CS 294-112, Deep
Reinforcement Learning, at UC Berkeley. See `cs294-112_stuff/homework.md` for
details. Here's the first figure they wanted:

![part01](figures/part_01.png?raw=true)

I think it looks good. Their "-300" requirement for the reward is a bit unclear.
The performance seems to reach that level but sometimes dips below. I ran this
with the default settings, except for increasing the number of iterations from
300 to 500.

Here's the next part, with the neural network value function:

![part02](figures/part_02.png?raw=true)

And here are the averages of these, so I took the average of the four linear
ones, and the four NN ones:

![part02_smoothed](figures/part_02_smooth.png?raw=true)

Ack, is the NN even that much better? It looks like it can explain the variance
slightly better (since, well, it *is* a neural network) but the performance is
not noticeably better. Maybe the problem isn't hard enough?

TODO for plotting and the actual code:

- Better analysis (not for code but for plotting), e.g. with error bars/regions.
- See what happens if the log stdev is the output of our network
- Also, what about using just tensorflow's basic Gaussians, not the
  contrib.distributions, which technically isn't supported.
- List the hyperparameters that I used.
- NPG, TRPO, etc.? I hope I can get to these.
