# Evolution Strategies

Inspired by [recent work from OpenAI][1].

# Code Usage

See the bash scripts.

The ES code I am using includes the following tricks:

- Mirrored sampling 
- Ranking transformation

I do not use the trick of instantiating a large block of Gaussian noise for each
worker, because this code is designed to run sequentially.

Note: as of 05/18/2017, `npop` INCLUDES the mirroring so it must be divisible by
two.

It uses TensorFlow but maybe that's not even needed for our purposes? Because
there are no gradients to update a network. (We have to do gradient ascent, but
that's done explicitly here and I don't think autodiff is necessary.) Tensorflow
and the GPU are mostly useful for the *forward* pass in RL, which is not even
the most critical step.


# Results

## Inverted Pendulum

I originally ran this for 800 iterations, but it seems like 700 is also a safe
upper bound on the number of iterations.

Args (this one is with `npop` **not** counting the mirroring):

```
Namespace(do_not_save=False, envname='InvertedPendulum-v1', es_iters=800,
log_every_t_iter=2, lrate_es=0.005, npop=100, render=False, seed=4, sigma=0.1,
snapshot_every_t_iter=50, test_trajs=10, verbose=True)
```

and (with revised `npop` and also with 700 iterations):

```
Namespace(do_not_save=False, envname='InvertedPendulum-v1', es_iters=700,
log_every_t_iter=2, lrate_es=0.005, npop=200, render=False, seed=5, sigma=0.1,
snapshot_every_t_iter=50, test_trajs=10, verbose=True)
```

The results look good! The results reach the perfect score *faster* than with
the Gaussian sampling of actions afterwards.

![InvertedPendulum01](figures/InvertedPendulum-v1_log.png?raw=true)

![InvertedPendulum02](figures/InvertedPendulum-v1_rewards_std.png?raw=true)


## Inverted Pendulum (+Gaussian Sampling)

This uses the Gaussian sampling, which we should *not* be doing.

Note: I ran this twice with normalized features (seeds 0 and 1), twice with
ranking transformation (seeds 2 and 3).

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

I then did seeds 2 and 3, which has the actual reward rank transformation. Seed
3 uses the following args (note: this time, `npop` gets doubled ... so they are
equivalent ... sorry for the confusion):

```
Namespace(do_not_save=False, envname='InvertedPendulum-v1', es_iters=1000,
log_every_t_iter=1, lrate_es=0.005, npop=100, render=False, seed=3, sigma=0.1,
snapshot_every_t_iter=50, test_trajs=10, verbose=True)
```

Yowza! It works! Here's the figure log, followed by the rewards and standard
deviations. They all reach the maximum score of 1000. I stopped seed 3 after 800
iterations, so **if I need to use expert trajectories, use seed 2** because that
one ran for the full 1000 iterations. The seed 2 trial took about 8 and 1/3
hours.

The plots named "Final"-something contain statistics related to the 10
rollouts the agent made each iteration, *after* it made the evolution strategies
weight update. The plots named "Scores"-something contain statistics related to
the 200 rollouts the agent made each iteration, where the 200 rollouts are each
with some weight perturbation. (This is the `npop` parameter I have.)

![InvertedPendulum01](figures/InvertedPendulum-v1-old_log.png?raw=true)

![InvertedPendulum02](figures/InvertedPendulum-v1-old_rewards_std.png?raw=true)


## Half Cheetah-v1 (+Gaussian Sampling)

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
