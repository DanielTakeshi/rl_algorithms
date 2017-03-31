# Deep Q-Networks

This is not entirely my code. It originally came from UC Berkeley's Deep
Reinforcement Learning class.  These are their comments:

> See http://rll.berkeley.edu/deeprlcourse/docs/hw3.pdf for instructions
> 
> The starter code was based on an implementation of Q-learning for Atari
> generously provided by Szymon Sidor from OpenAI

The rest of this README contains my comments and results. 

# Usage, Games, etc.

First, here is example usage:

```
python run_dqn_atari.py --seed 0 --log breakout_s000.pkl --num_timesteps 24000000 | tee text_logs/breakout_seed000.text
```

The statistics for plotting data will be stored in the `rewards.pkl` file inside
a `log` directory. I also like to save the stdout to inspect them for later.

Here's the `task` stuff in the code, ordered by index (i.e. 0, 1, etc.).

```
Task<env_id=BeamRiderNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=363.9 reward_ceiling=60000.0>
Task<env_id=BreakoutNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=1.7 reward_ceiling=800.0>
task=Task<env_id=EnduroNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=0.0 reward_ceiling=5000.0>
Task<env_id=PongNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=-20.7 reward_ceiling=21.0>
Task<env_id=QbertNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=163.9 reward_ceiling=40000.0>
```

The default for these is 40 million episodes, but that's not always needed for
the easier games.

TODO: get a mapping from string to task so I can put this in the argparse. I
should also use this for the plots to get the correct axes ranges. Put this in a
separate file ...

# Results

## Pong

Number of time steps I used: 32 million, resulting in roughly 7.5 million
"game steps." This is *not* the number of episodes ... sorry, it's confusing.

The results look good.

![pong](figures/pong.png?raw=true)

## Breakout

In progress ...
