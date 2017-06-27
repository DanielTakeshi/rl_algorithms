# Vanilla Policy Gradients

This is the standard vanilla policy gradients with stochastic policies, either
continuous or discrete. I based this off of CS 294-112 starter code.

I'm using Python 3.5.2 and Tensorflow 1.2.0. This code will not work with Python
2.7.x.  Note to self: when running bash scripts in GNU screen mode, be sure to
source my Python 3 conda environment.

# Simple Baselines

## CartPole-v0

Based on `bash_scripts/CartPole-v0.sh`:

![](figures/CartPole-v0.png?raw=true)

![](figures/CartPole-v0_sm.png?raw=true)

Architectures:

- **Policy**: (input) - 50 - (output), tanh
- **NN vf**: (input) - 50 - 50 - (output), tanh

## Pendulum-v0

Based on `bash_scripts/Pendulum-v0.sh`:

![Pendulum-v0](figures/Pendulum-v0.png?raw=true)

![Pendulum-v0_sm](figures/Pendulum-v0_sm.png?raw=true)

Architectures:

- **Policy**: (input) - 32 - 32 - (output), relu
- **NN vf**: (input) - 50 - 50 - (output), tanh

I think it looks OK. Pendulum is a bit tricky to solve because it requires an
adaptive learning rate but I still get close to about -100 or so. I'm not sure
what the theoretical best solution is; maybe zero, but that seems impossible.
The neural network is only slightly better with these results (I guess?) because
the problem is so simple. The action dimension is just one.



**TODO** haven't tested with these with new API ...
# MuJoCo Baselines

Tested on in alphabetical order:

- HalfCheetah-v1
- Hopper-v1
- Walker2d-v1

## HalfCheetah-v1

The raw runs based on `bash_scripts/halfcheetah.sh`:

![HalfCheetah-v1](figures/HalfCheetah-v1.png?raw=true)

And the smooth runs:

![HalfCheetah-v1_sm](figures/HalfCheetah-v1_sm.png?raw=true)

The GAIL paper said HalfCheetah-v1 should get around 4463.46 ± 105.83 and in
fact we are almost getting to that level. That's interesting.

What's confusing is that the explained variance for the linear case seems to be
terrible. Then why is the linear VF even working, and why is it just barely
worse than the NN value function? Hmmm ... I may want to catch a video of this
in action.

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

Next, Walker2d-v1. The raw runs based on `bash_scripts/walker.sh`:

![Walker2d-v1](figures/Walker2d-v1.png?raw=true)

And the smooth runs:

![Walker2d-v1_sm](figures/Walker2d-v1_sm.png?raw=true)

The GAIL paper said Walker-v1 should get around 6717.08 p/m 845.62, but that
might not be the same as Walker2d-v1. I'm not sure ... and the code Jonathan Ho
has for imitation learning doesn't do as well.
