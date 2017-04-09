"""
To plot this, you need to provide the experiment directory.
"""
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument("expdir", help="experiment dir, e.g., /tmp/experiments")
args = parser.parse_args()

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
from pylab import *
import os
from os.path import join
dirnames = os.listdir(args.expdir)

fig, axes = subplots(4, figsize=(12,10))
for dirname in dirnames:
    print(dirname)
    A = np.genfromtxt(join(args.expdir, dirname, 'log.txt'),delimiter='\t',dtype=None, names=True)
    # axes[0].plot(scipy.signal.savgol_filter(A['EpRewMean'] , 21, 3), '-x')
    x = A['TimestepsSoFar']
    axes[0].plot(x, A['EpRewMean'], '-x')
    axes[1].plot(x, A['KLOldNew'], '-x')
    axes[2].plot(x, A['Entropy'], '-x')
    axes[3].plot(x, A['EVBefore'], '-x')
legend(dirnames,loc='best').draggable()
axes[0].set_ylabel("EpRewMean")
axes[1].set_ylabel("KLOldNew")
axes[2].set_ylabel("Entropy")
axes[3].set_ylabel("EVBefore")
axes[3].set_ylim(-1,1)
axes[-1].set_xlabel("Iterations")
fig.savefig("figures/hw_part_1.png")
