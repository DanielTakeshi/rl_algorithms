"""
To plot this, you need to provide the experiment directory plus an output stem.
I use this for InvertedPendulum:

    python plot.py outputs/InvertedPendulum-v1 --envname InvertedPendulum-v1 \
            --out figures/InvertedPendulum-v1

(c) May 2017 by Daniel Seita
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
import sys
from os.path import join
from pylab import *
plt.style.use('seaborn-darkgrid')
sns.set_context(rc={'lines.markeredgewidth': 1.0})
np.set_printoptions(edgeitems=100,linewidth=100,suppress=True)
NUM_THREADS = 32

# Some matplotlib settings.
LOGDIR = 'outputs/'
FIGDIR = 'figures/'
title_size = 22
tick_size = 18
legend_size = 20
ysize = 18
xsize = 18
lw = 1
ms = 8
error_region_alpha = 0.3

# Attributes to include in a plot.
ATTRIBUTES = ["FinalAvgReturns",
              "FinalStdReturns",
              "FinalMaxReturns",
              "FinalMinReturns",
              "ScoresAvg",
              "ScoresStd",
              "ScoresMax",
              "ScoresMin"]

# Axes labels for environments.
ENV_TO_YLABELS = {"InvertedPendulum-v1": [0,1000]}


def plot_one_dir(args, directory):
    """ The actual plotting code.

    Assumes that we'll be plotting from one directory, which usually means
    considering one random seed only, however I'll see what happens. For ES, we
    should store the output at *every* timestep, so A['TotalIterations'] should
    be like np.arange(...), but this is more general in case we figure out a way
    to run for many more iterations.
    """
    print("Now plotting based on directory {} ...".format(directory))

    ### Figure 1: The log.txt file.
    A = np.genfromtxt(join(args.expdir, directory, 'log.txt'),
                      delimiter='\t', dtype=None, names=True)
    num = len(ATTRIBUTES)
    fig, axes = subplots(num, figsize=(12,3*num))
    x = A['TotalIterations']
    for (i,attr) in enumerate(ATTRIBUTES):
        axes[i].plot(x, A[attr], '-', lw=lw, color='darkred')
        axes[i].set_ylabel(attr, fontsize=ysize)
        axes[i].tick_params(axis='x', labelsize=tick_size)
        axes[i].tick_params(axis='y', labelsize=tick_size)
    plt.tight_layout()
    plt.savefig(args.out+'_log.png')

    ### Figure 2: Error regions.
    fig = plt.figure(figsize=(12,10))
    plt.plot(A['TotalIterations'], A["FinalAvgReturns"], 
             color='blue', marker='x', ms=ms, lw=lw)
    plt.fill_between(A['TotalIterations'],
                     A["FinalAvgReturns"] - A["FinalStdReturns"],
                     A["FinalAvgReturns"] + A["FinalStdReturns"],
                     alpha = error_region_alpha,
                     facecolor='y')
    plt.ylim(ENV_TO_YLABELS[args.envname])
    plt.title("Mean Episode Rewards (10 Trials)", fontsize=title_size)
    plt.xlabel("ES Iterations", fontsize=xsize)
    plt.ylabel("Rewards", fontsize=ysize)
    plt.tight_layout()
    plt.savefig(args.out+'_rewards_std.png')


if __name__ == "__main__":
    """ 
    Handle logic with argument parsing. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("expdir", help="experiment dir, e.g., /tmp/experiments")
    parser.add_argument("--out", type=str, help="full directory where to save")
    parser.add_argument("--envname", type=str)
    args = parser.parse_args()

    # For now ... should have many seeds later.
    assert len(os.listdir(args.expdir)) == 1 
    directory = os.listdir(args.expdir)[0]
    plot_one_dir(args, directory)
