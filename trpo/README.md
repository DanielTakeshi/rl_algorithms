# Trust Region Policy Optimization

Code outline:

- `main.py` sets up the options and the top-level call to TRPO.
- `trpo.py` contains the TRPO agent, describing how it gets paths, computes
  advantages, etc.
- `utils_trpo.py` contains two particular utils ...
- `fxn_approx.py` contains linear and neural network value functions.
