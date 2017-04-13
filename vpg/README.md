# Vanilla Policy Gradients

This repository started out as the homework assignment from CS 294-112, Deep
Reinforcement Learning, at UC Berkeley. See `cs294-112_stuff/homework.md` for
details. I then changed it to make it more general.

## Pendulum

Script I used:

```
#!/bin/bash
python main.py Pendulum-v0 --vf_type linear --seed 4
python main.py Pendulum-v0 --vf_type nn --seed 4
python main.py Pendulum-v0 --vf_type linear --seed 6
python main.py Pendulum-v0 --vf_type nn --seed 6
python main.py Pendulum-v0 --vf_type linear --seed 8
python main.py Pendulum-v0 --vf_type nn --seed 8
```

Here are the raw results:

![Pendulum-v0](figures/Pendulum-v0.png?raw=true)

And now the smoothed versions:

![Pendulum-v0_sm](figures/Pendulum-v0_sm.png?raw=true)

I think it looks OK. Pendulum is a bit tricky to solve because it requires an
adaptive learning rate but I still get close to zero (the theoretically optimal
solution, I think, but even TRPO can only get -100 or so). The neural network is
only slightly better because the problem is so simple. The action dimension is
just one.
