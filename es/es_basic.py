"""
Basic evolution strategies, based on Andrej Karpathy's starter code. We have the
actual solutions here only for didactic purposes, and to get the number of
weights correct. Actually, how similar is this to the cross entropy method that
I've worked with before? Both re-scale and then center at the new update. I
remember that the CEM may have only used a strict cutoff, but that's not really
a huge difference (we could smooth it out).

Tested on both solution sets here.
"""

import argparse
import sys
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import numpy as np
np.set_printoptions(suppress=True, precision=5)

# Adjust the solutions here as needed.
SOLUTIONS = [
    np.array([0.5, 0.1, -0.3]),
    np.array([0.5, 0.1, -0.3, 1.0, 1.0, 0.4])
]


def f(w, sol):
    """ Define whatever function we want to optimize. """
    return -np.sum(np.square(sol-w))


def run_es(args):
    """ Runs basic evolution strategies using the provided arguments. 
    
    It uses the solution here only for the size and determining performance. In
    a real problem, we wouldn't have access to it. Also, we standardize the
    rewards. I think just because it makes sense to standardize a lot of things
    and in our case we're just concerned about the relative benefit for each
    weight, so rescaling is OK (e.g. for numerical purposes).

    We adjust the amount of noise with sigma. This is the standard deviation and
    we multiply it with the standard Gaussian, which is the usual way to do it.

    The last line performs something that _looks_ like a gradient update. What
    really happens is that each weight is updated using a linear combination of
    all the weights, plus some jittered noise. We divide by npop because
    otherwise we'd get super-large changes.
    """

    sol = SOLUTIONS[args.sol_index]
    w = np.random.randn(sol.size)

    for i in range(args.num_iters):
        if (i % args.print_every == 0):
            print("iter {}.  w: {},  solution: {},  reward: {:.5f}".format(
                    str(i).zfill(4), str(w), sol, f(w,sol)))
        N = np.random.randn(args.npop, sol.size)
        R = np.zeros(args.npop)
        for j in range(args.npop):
            R[j] = f(w+args.sigma*N[j], sol)
        A = (R - np.mean(R)) / (np.std(R)+0.000001)
        w = w + args.lrate/(args.npop*args.sigma) * np.dot(N.T, A)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--npop', type=int, default=50)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--sol_index', type=int, default=0)
    parser.add_argument('--num_iters', type=int, default=500)
    parser.add_argument('--print_every', type=int, default=20)
    args = parser.parse_args()
    run_es(args)
