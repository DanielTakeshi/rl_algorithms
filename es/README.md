# Evolution Strategies

Inspired by [recent work from OpenAI][1].

# Code Usage

See the bash scripts.

The ES code I am using includes the following tricks:

- Mirrored sampling
- Ranking transformation

I do not use the trick of instantiating a large block of Gaussian noise for each
worker, because this code is designed to run sequentially.

It uses TensorFlow but maybe that's not even needed for our purposes? Because
there are no gradients to update a network. (We have to do gradient ascent, but
that's done explicitly here and I don't think autodiff is necessary.) Tensorflow
and the GPU are mostly useful for the *forward* pass in RL, which is not even
the most critical step.


# Results

## Inverted Pendulum

Note: this was with normalized features, not the ranking transformation.

Run with (for one seed):

```
rm -r outputs/InvertedPendulum-v1/seed0000
clear
python main.py InvertedPendulum-v1 \
    --es_iters 1000 \
    --log_every_t_iter 1 \
    --npop 200 \
    --seed 0 \
    --sigma 0.1 \
    --test_trajs 10 \
    --verbose
```

And then do the same thing, but with seed 0001 instead.

I then did seed 3, which has the actual reward rank transformation. It uses the
following args (note: this time, `npop` gets doubled ... so they are equivalent
... sorry for the confusion):

```
Namespace(do_not_save=False, envname='InvertedPendulum-v1', es_iters=1000,
log_every_t_iter=1, lrate_es=0.005, npop=100, render=False, seed=3, sigma=0.1,
snapshot_every_t_iter=50, test_trajs=10, verbose=True)
```

Yowza! It works! Here's the figure log, followed by the rewards and standard
deviations. They all reach the maximum score of 1000.

The plots named "Final"-something contain statistics related to the 10
rollouts the agent made each iteration, *after* it made the evolution strategies
weight update. The plots named "Scores"-something contain statistics related to
the 200 rollouts the agent made each iteration, where the 200 rollouts are each
with some weight perturbation. (This is the `npop` parameter I have.)

![InvertedPendulum01](figures/InvertedPendulum-v1_log.png?raw=true)

![InvertedPendulum02](figures/InvertedPendulum-v1_rewards_std.png?raw=true)


## Half Cheetah-v1

Run with:

```
rm -r outputs/HalfCheetah-v1/seed0000
clear
python main.py HalfCheetah-v1 \
    --es_iters 1000 \
    --log_every_t_iter 1 \
    --npop 200 \
    --seed 0 \
    --sigma 0.1 \
    --test_trajs 10 \
    --verbose
```

Actually, I had to terminate this one after about 690 iterations because it was
taking too long. This took maybe 16 hours to generate! But at least it is
learning *something*. TRPO on HalfCheetah-v1 gets roughly 2400 so this has a
long way to go.

![HalfCheetah01](figures/HalfCheetah-v1_log.png?raw=true)

![HalfCheetah02](figures/HalfCheetah-v1_rewards_std.png?raw=true)


[1]:https://blog.openai.com/evolution-strategies/
