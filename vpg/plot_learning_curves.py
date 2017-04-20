"""
To plot this, you need to provide the experiment directory. I did this on my
laptop for the "homework"-related stuff:

python plot_learning_curves.py outputs/part01/ --out figures/part_01.png
python plot_learning_curves.py outputs/part02/ --out figures/part_02.png
python plot_learning_curves.py outputs/part02/ --out figures/part_02_smooth.png --smooth

For the refactored, generic vanilla policy gradient code, do (after first
checking niter in this code...):

python plot_learning_curves.py outputs/Pendulum-v0 --out figures/Pendulum-v0.png
python plot_learning_curves.py outputs/Pendulum-v0 --out figures/Pendulum-v0_sm.png --smooth

(Don't forget to add `sm` to the figure name!)

Do this for each environment tested, e.g. with Hopper-v1 as well. Also, be
careful to check the number of iterations!
"""

import argparse
import os
from os.path import join
import sys
from pylab import *
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# Argument parser.
parser = argparse.ArgumentParser()
parser.add_argument("expdir", help="experiment dir, e.g., /tmp/experiments")
parser.add_argument("--out", type=str, help="full directory where to save")
parser.add_argument("--niter", type=int, default=100, help="the number of iterations used")
parser.add_argument('--smooth', action='store_true')
args = parser.parse_args()
dirnames = os.listdir(args.expdir)
niter = args.niter

# CAREFUL!
if 'Pendulum-v0' in args.expdir:
    niter = 500
if ('Hopper-v1' in args.expdir) or ('Walker2d-v1' in args.expdir):
    niter = 3000
print("dirnames:\n{}".format(dirnames))
print("niter: {}".format(niter))

# Matplotlib settings
lw=2
font = 18
fig, axes = subplots(4, figsize=(14,18))

# Try to handle the smoothed case separately. It's a bit ugly. I'm assuming I
# did three different seeds, BTW.
if args.smooth:
    n = 3
    colors = ['gold', 'midnightblue']

    avgx_lin = np.zeros(niter,)
    avgx_nn = np.zeros(niter,)
    avg_rew_lin = np.zeros(niter,)
    avg_rew_nn = np.zeros(niter,)
    avg_kl_lin = np.zeros(niter,)
    avg_kl_nn = np.zeros(niter,)
    avg_ent_lin = np.zeros(niter,)
    avg_ent_nn = np.zeros(niter,)
    avg_ev_lin = np.zeros(niter,)
    avg_ev_nn = np.zeros(niter,)

    for dirname in dirnames:
        A = np.genfromtxt(join(args.expdir, dirname, 'log.txt'),delimiter='\t',dtype=None, names=True)
        if 'linearvf' in dirname:
            avgx_lin += A['TimestepsSoFar']
            avg_rew_lin += A['EpRewMean']
            avg_kl_lin += A['KLOldNew']
            avg_ent_lin += A['Entropy']
            avg_ev_lin += A['EVBefore']
        elif 'nnvf' in dirname:
            avgx_nn += A['TimestepsSoFar']
            avg_rew_nn += A['EpRewMean']
            avg_kl_nn += A['KLOldNew']
            avg_ent_nn += A['Entropy']
            avg_ev_nn += A['EVBefore']

    avgx_lin /= n
    avgx_nn /= n
    avg_rew_lin /= n
    avg_rew_nn /= n
    avg_kl_lin /= n
    avg_kl_nn /= n
    avg_ent_lin /= n
    avg_ent_nn /= n
    avg_ev_lin /= n
    avg_ev_nn /= n

    axes[0].plot(avgx_lin, avg_rew_lin, '-', color=colors[0], lw=lw)
    axes[0].plot(avgx_nn,  avg_rew_nn,  '-', color=colors[1], lw=lw)
    axes[1].plot(avgx_lin, avg_kl_lin,  '-', color=colors[0], lw=lw)
    axes[1].plot(avgx_nn,  avg_kl_nn,   '-', color=colors[1], lw=lw)
    axes[2].plot(avgx_lin, avg_ent_lin, '-', color=colors[0], lw=lw)
    axes[2].plot(avgx_nn,  avg_ent_nn,  '-', color=colors[1], lw=lw)
    axes[3].plot(avgx_lin, avg_ev_lin,  '-', color=colors[0], lw=lw, label='Linear VF')
    axes[3].plot(avgx_nn,  avg_ev_nn,   '-', color=colors[1], lw=lw, label='NN VF')
    axes[3].legend(loc='best',ncol=2)

else:
    colors = ['blue', 'red', 'yellow', 'black', 'gray', 'cyan']
    for dirname, c in zip(dirnames, colors):
        A = np.genfromtxt(join(args.expdir, dirname, 'log.txt'),delimiter='\t',dtype=None, names=True)
        x = A['TimestepsSoFar']
        axes[0].plot(x, A['EpRewMean'], '-', color=c, lw=lw)
        axes[1].plot(x, A['KLOldNew'], '-', color=c, lw=lw)
        axes[2].plot(x, A['Entropy'], '-', color=c, lw=lw)
        axes[3].plot(x, A['EVBefore'], '-', color=c, lw=lw)
        legend(dirnames,loc='best',ncol=2).draggable()

axes[0].set_ylabel("EpRewMean", fontsize=font)
axes[1].set_ylabel("KLOldNew", fontsize=font)
axes[2].set_ylabel("Entropy", fontsize=font)
axes[3].set_ylabel("EVBefore", fontsize=font)
axes[3].set_ylim(-1,1)
axes[-1].set_xlabel("Iterations", fontsize=font)
fig.savefig(args.out)
