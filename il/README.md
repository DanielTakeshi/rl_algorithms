# Imitation Learning

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

The following contain my notes, observations, and results.


# Commands, Code, etc.

Use:

```
python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --render --num_rollouts 20
```

These generate expert policies and I have modified the code to save these. I can
benchmark the rollout results with those in the [Generative Adversarial
Imitation Learning paper][1] to ensure that performance is comparable. (See
Table 1 in that paper.)

# Results

In progress ..



[1]:https://arxiv.org/abs/1606.03476
