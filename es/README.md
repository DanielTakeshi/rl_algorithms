# Evolution Strategies

Inspired by [recent work from OpenAI][1].

# Code Usage

In progress ...


# Results

## Inverted Pendulum

Run with:

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

And then do the same thing, but with seed 0001 instead. (Note: I need to start
printing out the entire `args` from `argparse` and then saving that somewhere
instead of relying on bash scripts for settings.)

Yowza! It works! Here's the figure log, followed by the rewards and standard
deviations. Both seeds exhibit similar behavior, with seed 0001 reaching the
maximum score of 1000 slightly sooner.

The plots named "Final"-something contain statistics related to the 10
rollouts the agent made each iteration, *after* it made the evolution strategies
weight update. The plots named "Scores"-something contain statistics related to
the 200 rollouts the agent made each iteration, where the 200 rollouts are each
with some weight perturbation. (This is the `npop` parameter I have.)

![InvertedPendulum01](figures/InvertedPendulum-v1_log.png?raw=true)

![InvertedPendulum02](figures/InvertedPendulum-v1_rewards_std.png?raw=true)

[1]:https://blog.openai.com/evolution-strategies/
