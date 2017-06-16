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
titlesize = 25
LRATES = [0.0005, 0.001, 0.005, 0.01, 0.05]
REGUS  = [0.0005, 0.001, 0.005, 0.01, 0.05]


def get_info_from_name(string):
    """ Gets information I need from the file name.
    
    Removing last 4 removes the .npy extension. Returns: name, rollouts,
    trainiters, batch size, lrate, and reg param. We don't need all of these but
    it's easiest if no indices are missing. 
    """
    sp = string[:-4].split('_')
    return (sp[0], sp[1], sp[2], sp[3], sp[4], sp[5])


def plot_bc(args):
    """ Plot the results. Yeah, it's a big ugly ... """

    # Load the data (numpy stuff).
    files = {}
    for f in os.listdir(RESULTS_DIR):
        if args.envname in f:
            files[f] = (np.load(RESULTS_DIR+f))[()]

    # Now plot and save w.r.t. different rollouts. I'm finally going to use this
    # method of automating: save the indices in a dictionary and then just get
    # the i,j from that, rather than looping through them.
    fig, axarr = plt.subplots(2,2, figsize=(15,12))
    nums_dict = {'0004':(0,0),  '0011':(0,1), '0018':(1,0), '0025':(1,1)}

    for numstr in nums_dict: 
        i,j = nums_dict[numstr]
        fs = sorted([k for k in files if numstr in k])
        means = [files[s]['mean'] for s in fs]
        axarr[i,j].set_title("{} {} Rollouts".format(args.envname,int(numstr)),
                             fontsize=titlesize)
        for lr in LRATES:
            fnames = [n for n in fs if get_info_from_name(n)[4] == str(lr)]
            means = [files[s]['mean'] for s in fnames]
            axarr[i,j].plot(REGUS, means, lw=3, label="lrate = {}".format(lr))
        axarr[i,j].legend(loc='lower left')
        axarr[i,j].set_xscale('log')
        axarr[i,j].set_xlim([0.00005,0.1])
        axarr[i,j].set_ylim([0,2000])

    plt.tight_layout()
    plt.savefig(FIG_DIR+args.envname+ ".png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    args = parser.parse_args()
    plot_bc(args)
    print("All done!")
