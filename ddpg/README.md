# Deep Deterministic Policy Gradients

- Python 3.5
- Tensorflow 1.2

I'm following the original DDPG paper as much as possible, and using their
"low-dimensional" representation, not the pixels-based one.

## Pendulum-v0

Action space: -2 to 2.

```
python main.py Pendulum-v0
```

Status: not yet working. I think it's done but alas there is some bug somewhere.
Ugh.


## References

(These might be useful to supplement the original paper.)

- http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
- https://github.com/rmst/ddpg
- https://github.com/openai/rllab
- https://github.com/yukezhu/tensorflow-reinforce
- https://github.com/stevenpjg/ddpg-aigym
