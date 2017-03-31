# Deep Q-Networks

The starter code is from UC Berkeley's Deep Reinforcement Learning class.  These
are their comments:

> See http://rll.berkeley.edu/deeprlcourse/docs/hw3.pdf for instructions
> 
> The starter code was based on an implementation of Q-learning for Atari
> generously provided by Szymon Sidor from OpenAI

The rest of this README contains my comments and results. 

# Usage, Games, etc.

First, here is example usage (slashes are only for readability here):

```
python run_dqn_atari.py --game Pong --seed 1 --num_timesteps 30000000 | tee logs_text/Pong_s001.text
```

With these settings, the statistics for plotting data will be stored in the
`log_pkls/Pong_s001.pkl` file.

Here are some of the `task` stuff in the code, ordered by index (i.e. 0, 1,
etc.).

```
Task<env_id=BeamRiderNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=363.9 reward_ceiling=60000.0>
Task<env_id=BreakoutNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=1.7 reward_ceiling=800.0>
Task<env_id=EnduroNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=0.0 reward_ceiling=5000.0>
Task<env_id=PongNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=-20.7 reward_ceiling=21.0>
Task<env_id=QbertNoFrameskip-v3 trials=2 max_timesteps=40000000 max_seconds=None reward_floor=163.9 reward_ceiling=40000.0>
```

The default for these is 40 million episodes, but that's not always needed for
the easier games.

The `num_timesteps` parameter corresponds to the number of steps in the
"underlying" environment, *not* the "wrapped" environment. See the stopping
criterion:

```
def stopping_criterion(env, t):
    # notice that here t is the number of steps of the wrapped env,
    # which is different from the number of steps in the underlying env
    return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps
```

The `t` here is what I think of as "the number of steps." It's confusing, I
know. There might be a better way to handle this. From now on, when I say
"steps", it refers to the `t`-like number, and *not* the `num_timesteps`
parameter.

# Results

Notes:

- Timing results are based on running with an NVIDIA Titan X with Pascal GPU.
- Scores per Episode indicate scores for every episode.
- Scores per Timestep indicate the score of the current episode at a given
  timestep; each episode requires some number of timesteps for the agent to
  complete it. Theres' a lot of them, so I take every 10,000.
- Blocks mean taking an interval of some size (100) and taking the mean.

## Pong

- `num_timesteps`: 32 million
- Training steps: about 7.5 million
- Time: about 9.5 hours.

![pong](figures/Pong.png?raw=true)

## Breakout

- `num_timesteps`: 24 million
- Training steps: about 5.7 million
- Episodes: about 10000.
- Time: about 6.9 hours.

Yeah, I have *no idea* why the performance just dropped like that ... let me run
again with a different seed.

![breakout](figures/Breakout.png?raw=true)
