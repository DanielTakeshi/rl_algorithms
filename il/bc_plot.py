"""
(c) April 2017 by Daniel Seita

Code for plotting behavioral cloning.
"""

import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
plt.style.use('seaborn-darkgrid')
np.set_printoptions(edgeitems=100,
                    linewidth=100,
                    suppress=True)
FIG_DIR = 'figures/'
RESULTS_DIR = 'results/'


def get_info_from_name(string):
    """ Get information I need from string. Removing last 4 removes the .npy
    extension. Returns: rollouts, trainiters, lrate, and reg param. """
    sp = string.split('_')
    return sp[1], sp[2], sp[4], sp[5]


def plot_bc(args):
    """ Plot the results. """

    # Load the data.
    files = {}
    for f in os.listdir(RESULTS_DIR):
        if args.envname in f:
            files[f] = (np.load(RESULTS_DIR+f))[()]

    # Now plot and save w.r.t. different rollouts. I'll figure out a better way
    # to do this once I find the time.
    fig, axarr = plt.subplots(2,2, figsize=(15,12))

    fs = [k for k in files if '0004' in k]
    means = [files[s]['mean'] for s in fs]
    print(means)
    axarr[0,0].set_title(args.envname+ " 4 Rollouts")
    axarr[0,0].plot(means)

    fs = [k for k in files if '0011' in k]
    means = [files[s]['mean'] for s in fs]
    axarr[0,1].set_title(args.envname+ " 11 Rollouts")
    axarr[0,1].plot(means)

    fs = [k for k in files if '0018' in k]
    means = [files[s]['mean'] for s in fs]
    axarr[1,0].set_title(args.envname+ " 18 Rollouts")
    axarr[1,0].plot(means)

    fs = [k for k in files if '0025' in k]
    means = [files[s]['mean'] for s in fs]
    axarr[1,1].set_title(args.envname+ " 25 Rollouts")
    axarr[1,1].plot(means)

    plt.tight_layout()
    plt.savefig(FIG_DIR+args.envname+ ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    args = parser.parse_args()
    plot_bc(args)
    print("All done!")
