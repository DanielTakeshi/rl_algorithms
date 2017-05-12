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

Yowza! It works! Here's the figure log, followed by the rewards and standard
deviations.

![InvertedPendulum01](figures/InvertedPendulum-v1_log.png?raw=true)

![InvertedPendulum02](figures/InvertedPendulum-v1_rewards_std.png?raw=true)

[1]:https://blog.openai.com/evolution-strategies/
