# Deep Q-Networks

This is not my code; it is from UC Berkeley's Deep Reinforcement Learning class.
These are their comments:

> See http://rll.berkeley.edu/deeprlcourse/docs/hw3.pdf for instructions
> 
> The starter code was based on an implementation of Q-learning for Atari
> generously provided by Szymon Sidor from OpenAI

The rest of this README contains my comments and results. First, here is the
usage:

```
python run_dqn_atari.py --seed 0 --log rewards.pkl
```

The names should be straightforward. The statistics for plotting data will be
stored in the `rewards.pkl` file inside a `log` directory.
