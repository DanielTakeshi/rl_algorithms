"""
To plot this, you need to provide the experiment directory. For the first part,
which is just plotting performnace, do (at least on my laptop):

python plot_learning_curves.py outputs/part01/ --out figures/part_01.png

Next part:

python plot_learning_curves.py outputs/part02/ --out figures/part_02.png
python plot_learning_curves.py outputs/part02/ --out figures/part_02_smooth.png --smooth
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
parser.add_argument('--smooth', action='store_true')
args = parser.parse_args()
dirnames = os.listdir(args.expdir)
print("dirnames:\n{}".format(dirnames))

# Matplotlib settings
lw=2
fig, axes = subplots(4, figsize=(12,10))


# Try to handle the smoothed case separately. It's a bit ugly.
if args.smooth:
    niter = 500
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
    for dirname in dirnames:
        A = np.genfromtxt(join(args.expdir, dirname, 'log.txt'),delimiter='\t',dtype=None, names=True)
        x = A['TimestepsSoFar']
        axes[0].plot(x, A['EpRewMean'], '-', lw=lw)
        axes[1].plot(x, A['KLOldNew'], '-', lw=lw)
        axes[2].plot(x, A['Entropy'], '-', lw=lw)
        axes[3].plot(x, A['EVBefore'], '-', lw=lw)
        legend(dirnames,loc='best',ncol=2).draggable()

axes[0].set_ylabel("EpRewMean")
axes[1].set_ylabel("KLOldNew")
axes[2].set_ylabel("Entropy")
axes[3].set_ylabel("EVBefore")
axes[3].set_ylim(-1,1)
axes[-1].set_xlabel("Iterations")
fig.savefig(args.out)
