# Behavioral Cloning

## Main Idea

This runs Behavioral Cloning (BC) on MuJoCo environments, with settings inspired
by the NIPS 2016 [GAIL paper][1]. Specifically:

- The expert is TRPO and provided from Jonathan Ho (see below).

- Dataset consists of inputs (states=s) to labels (actions=a). They're
  continuous, so minimize mean L2 loss across minibatches. It is split into 70%
  training, 30% validation.

- Expert data size is measured in terms of the number of expert rollouts we
  collected. Note that, like the GAIL paper, I *subsample*, so the actual amount
  of (s,a) pairs available for BC is much smaller.

- The neural network (from TensorFlow) is fully connected and has two hidden
  layers of 100 units each with hyperbolic tangent non-linearities. It's trained
  with Adam, with a step size of 1e-3, and has a batch size of 128.

- For now, I plot validation set performance (i.e. loss) without really using
  it. If I needed to be formal and had to pick a given iteration for which to
  choose my BC expert (because different iterations mean different weights) I'd
  choose the one with best validation set performance. I also plot training set
  performance just for kicks.


## Running the Code

To run BC, there are several steps:

- (If needed) generate expert data. Run

  ```
  ./bash_scripts/gen_exp_data.sh
  ```

  which will run and save expert trajectories in numpy arrays. They're not saved
  in the repository (ask if you want my version). By default, the number of
  trajectories is saved into the file name by default and matches the values in
  the GAIL paper (see Table 1). 
  
- See the bash scripts for examples of running BC, such as
  `runbc_modern_stochastic.sh` which runs four MuJoCo environments for different
  expert dataset sizes random seeds.

- To plot the code, it's simple: `python plot_bc.py`. No command line arguments!


If you're interested:

- The expert performance that I'm seeing is roughly similar to what's reported
  in the GAIL paper, with the exception of the Walker environment, but I may be
  running a newer version from Jonathan Ho. See the output in
  `logs/gen_exp_data.text` for details.

- The `bash_scripts` directory also contains a file called `demo.bash`, which
  you can use to visualize expert trajectories, just for fun.


# Results

Note that Ant-v1, HalfCheetah-v1, Hopper-v1, and Walker2d-v1 use 4, 11, 18, and
25 expert rollouts since that follows the GAIL paper. Humanoid-v1 uses 80, 160,
and 240 expert trajectories.

Observations:

- BC does well in Ant-v1 and HalfCheetah-v1. 

- Hopper-v1 seems to be difficult, surprisingly. It has a relatively small state
  space compared to Ant (11 vs 111).

- Walker2d-v1 seems to be in between, BC doesn't get going until 25 rollouts.

## Ant-v1

![ant](figures/Ant-v1.png?raw=true)

## HalfCheetah-v1

![halfcheetah](figures/HalfCheetah-v1.png?raw=true)

## Hopper-v1

![hopper](figures/Hopper-v1.png?raw=true)

## Walker2d-v1

![walker2d](figures/Walker2d-v1.png?raw=true)


# Original Notes from Berkeley

This started from UC Berkeley's Deep Reinforcement Learning class. Here's their
information:

> Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym
> 
> The only file that you need to look at is `run_expert.py`, which is code to
> load up an expert policy, run a specified number of roll-outs, and save out
> data.
> 
> In `experts/`, the provided expert policies are:
> * Ant-v1.pkl
> * HalfCheetah-v1.pkl
> * Hopper-v1.pkl
> * Humanoid-v1.pkl
> * Reacher-v1.pkl
> * Walker2d-v1.pkl
> 
> The name of the pickle file corresponds to the name of the gym environment.

[1]:https://arxiv.org/abs/1606.03476
