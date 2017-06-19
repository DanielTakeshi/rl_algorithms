"""
(c) April 2017 by Daniel Seita

Code for plotting behavioral cloning. No need to use command line arguments,
just run `python plot_bc.py`. Easy! Right now it generates two figures per
environment, one with validation set losses and the other with returns. The
latter is probably more interesting.

TODO right now it really assumes we did 4, 11, 18, 25 ... really have to change
that and get it to work for Humanoid (for instance).
"""

import argparse
import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
np.set_printoptions(edgeitems=100, linewidth=100, suppress=True)

# Some matplotlib settings.
plt.style.use('seaborn-darkgrid')
error_region_alpha = 0.25
LOGDIR = 'logs/'
FIGDIR = 'figures/'
title_size = 22
tick_size = 17
legend_size = 17
ysize = 18
xsize = 18
lw = 3
ms = 8
colors = ['red', 'blue', 'yellow', 'black']

# TODO make this programmatic ...
ROLLOUTS = ['4', '11', '18', '25']
R_TO_IJ = {'4':(0,2), '11':(1,0), '18':(1,1), '25':(1,2)}

def plot_bc(edir):
    """ Plot the results for this particular environment. """
    subdirs = os.listdir(LOGDIR+edir)
    print("plotting subdirs {}".format(subdirs))

    # Make it easy to count how many of each numrollouts we have.
    R_TO_COUNT = {'4':0, '11':0, '18':0, '25':0}
    fig,axarr = plt.subplots(2, 3, figsize=(24,15))

    # A bunch of boring stuff.
    axarr[0,0].set_title(edir+", Training Losses", fontsize=title_size)
    axarr[0,1].set_title(edir+", Validation Losses", fontsize=title_size)
    axarr[0,2].set_title(edir+", Returns, " +ROLLOUTS[0]+ " Rollouts", fontsize=title_size)
    axarr[1,0].set_title(edir+", Returns, " +ROLLOUTS[1]+ " Rollouts", fontsize=title_size)
    axarr[1,1].set_title(edir+", Returns, " +ROLLOUTS[2]+ " Rollouts", fontsize=title_size)
    axarr[1,2].set_title(edir+", Returns, " +ROLLOUTS[3]+ " Rollouts", fontsize=title_size)

    # Don't forget to plot the expert performance here (really have to make this
    # more general...).
    exp04 = np.mean(np.load("expert_data/"+edir+"_004.npy")[()]['returns'])
    exp11 = np.mean(np.load("expert_data/"+edir+"_011.npy")[()]['returns'])
    exp18 = np.mean(np.load("expert_data/"+edir+"_018.npy")[()]['returns'])
    exp25 = np.mean(np.load("expert_data/"+edir+"_025.npy")[()]['returns'])
    axarr[0,2].axhline(y=exp04, color='brown', lw=lw, linestyle='--', label='expert')
    axarr[1,0].axhline(y=exp11, color='brown', lw=lw, linestyle='--', label='expert')
    axarr[1,1].axhline(y=exp18, color='brown', lw=lw, linestyle='--', label='expert')
    axarr[1,2].axhline(y=exp25, color='brown', lw=lw, linestyle='--', label='expert')

    for dd in subdirs:
        ddsplit = dd.split("_") # `dd` is of the form `numroll_X_seed_Y`
        numroll, seed = ddsplit[1], ddsplit[3]
        xcoord   = np.load(LOGDIR+edir+"/"+dd+"/iters.npy")
        tr_loss  = np.load(LOGDIR+edir+"/"+dd+"/tr_loss.npy")
        val_loss = np.load(LOGDIR+edir+"/"+dd+"/val_loss.npy")
        returns  = np.load(LOGDIR+edir+"/"+dd+"/returns.npy")
        mean_ret = np.mean(returns, axis=1)
        std_ret  = np.std(returns, axis=1)

        # Playing with dictionaries
        ijcoord = R_TO_IJ[numroll]
        cc = colors[ R_TO_COUNT[numroll] ]
        R_TO_COUNT[numroll] += 1

        axarr[ijcoord].plot(xcoord, mean_ret, lw=lw, color=cc, label=dd)
        axarr[ijcoord].fill_between(xcoord, 
                mean_ret-std_ret,
                mean_ret+std_ret,
                alpha=error_region_alpha,
                facecolor=cc)
        axarr[0,0].plot(xcoord, tr_loss, lw=lw, label=dd)
        axarr[0,1].plot(xcoord, val_loss, lw=lw, label=dd)

    for i in range(2):
        for j in range(3):
            if i == 0 and j == 0:
                axarr[i,j].set_ylabel("Loss Training MBs", fontsize=ysize)
            if i == 0 and j == 1:
                axarr[i,j].set_ylabel("Loss Validation Set", fontsize=ysize)
            else:
                axarr[i,j].set_ylabel("Average Return", fontsize=ysize)
            axarr[i,j].set_xlabel("Training Minibatches", fontsize=xsize)
            axarr[i,j].tick_params(axis='x', labelsize=tick_size)
            axarr[i,j].tick_params(axis='y', labelsize=tick_size)
            axarr[i,j].legend(loc="best", prop={'size':legend_size})
            axarr[i,j].legend(loc="best", prop={'size':legend_size})

    axarr[0,0].set_yscale('log')
    axarr[0,1].set_yscale('log')
    plt.tight_layout()
    plt.savefig(FIGDIR+edir+".png")


if __name__ == "__main__":
    env_dirs = [e for e in os.listdir(LOGDIR) if "text" not in e]
    print("Plotting with one figure per env_dirs = {}".format(env_dirs))
    for e in env_dirs:
        plot_bc(e)
