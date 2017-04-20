# Vanilla Policy Gradients

This repository started out as the homework assignment from CS 294-112, Deep
Reinforcement Learning, at UC Berkeley. See `cs294-112_stuff/homework.md` for
details. I then changed it to make it more general.

## Pendulum-v0

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
adaptive learning rate but I still get close to about -100 or so. I'm not sure
what the theoretical best solution is; maybe zero, but that seems impossible.
The neural network is only slightly better with these results because the
problem is so simple. The action dimension is just one.


## Hopper-v1

I used the script in `bash_scripts/hopper.sh`. Here are the raw results:

![Hopper-v1](figures/Hopper-v1.png?raw=true)

And now the smoothed versions:

![Hopper-v1_sm](figures/Hopper-v1_sm.png?raw=true)

(Ho & Ermon 2016) showed in the GAIL paper that Hopper-v1 should get 3571.38
with a standard deviation of 184.20 so ... yeah, these results are a bit
sub-par! But at least they are learning *something*. Maybe my version of TRPO
will do better.


## Walker2d-v1

Next, Walker2d-v1.

![Walker2d-v1](figures/Walker2d-v1.png?raw=true)

The GAIL paper said Walker-v1 shoudl get around 6717.08 p/m 845.62, but that
might not be the same as Walker2d-v1. I'm not sure ... and the code Jonathan Ho
has for imitaiton learning doesn't do as well.
