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

To generate expert data, run

```
bash bash_scripts/gen_exp_data.sh
```

which will run and save expert trajectories. They're not in this repository
since they're rather larger. The number of trajectories used should match the
numbers reported in the [Generative Adversarial Imitation Learning paper][1]
(see Table 1). The expert performance that I'm seeing is roughly similar to
what's reported in the paper, with the exception of the Walker environment, but
I may be running a different version of it. See the output in
`logs/gen_exp_data.text` for details.

To run behavioral cloning and to test results, use the `bash_scripts` directory
again, this time for scripts running the behavioral cloning code.

# Results

For Hopper, they're a bit odd, not sure why Ho & Ermon report such different
values. I'll need to check their loss function and also to check if the expert
policies are roughly the same dataset-wise as ours. This is what I see:

![hopper](figures/figures/Hopper-v1.png?raw=true)

Yeah ... why does the best use 11 rollouts? Beats me.

[1]:https://arxiv.org/abs/1606.03476
