import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
plt.style.use('seaborn-darkgrid')
np.set_printoptions(edgeitems=100,
                    linewidth=100,
                    suppress=True)

# Some default settings.
LOGDIR = 'logs_pkls/'
FIGDIR = 'figures/'
title_size = 22
axis_size = 20
tick_size = 18
legend_size = 18
lw = 3

def smoothed_block(x, n):
    """ Smoothed average in a block (size n) on array x. """
    assert (len(x.shape) == 1) and (x.shape[0] > n)
    length = x.shape[0]-n+1
    sm = np.zeros((length,))
    for i in range(length):
        sm[i] = np.mean(x[i:i+n])
    return sm


########
# Pong #
########

name = "Pong"
with open(LOGDIR+'Pong_s001.pkl', 'rb') as f:
    pong_data = np.array(pickle.load(f))
    pong_eps = np.array(pickle.load(f))
pong_data      = np.maximum(pong_data, -21)
pong_t         = (pong_data[:,0]) / 1000000.0
pong_mean      = pong_data[:,1]
pong_best_mean = pong_data[:,2]
pong_ep        = pong_data[:,3]
pong_eps_sm    = smoothed_block(pong_eps, 100)

fig, axarr = plt.subplots(2,2, figsize=(15,12))
axarr[0,0].set_title(name+ " Scores at Timesteps", fontsize=title_size)
axarr[0,1].set_title(name+ " Scores at Timesteps (Block 100)", fontsize=title_size)
axarr[0,0].plot(pong_t, pong_ep, c='red', lw=lw, label='Seed 0')
axarr[0,1].plot(pong_t, pong_mean, c='red', lw=lw, label='Seed 0')
axarr[0,0].set_xlabel("Training Steps (in Millions)", fontsize=axis_size)
axarr[0,1].set_xlabel("Training Steps (in Millions)", fontsize=axis_size)

axarr[1,0].set_title(name+ " Scores per Episode", fontsize=title_size)
axarr[1,1].set_title(name+ " Scores per Episode (Block 100)", fontsize=title_size)
axarr[1,0].plot(pong_eps, c='red', lw=lw, label='Seed 0')
axarr[1,1].plot(pong_eps_sm, c='red', lw=lw, label='Seed 0')
axarr[1,0].set_xlabel("Number of Episodes", fontsize=axis_size)
axarr[1,1].set_xlabel("Number of Episodes", fontsize=axis_size)

for i in range(2):
    for j in range(2):
        axarr[i,j].set_ylabel("Rewards", fontsize=axis_size)
        axarr[i,j].tick_params(axis='x', labelsize=tick_size)
        axarr[i,j].tick_params(axis='y', labelsize=tick_size)
        axarr[i,j].legend(loc='lower right', prop={'size':legend_size})
        axarr[i,j].set_ylim([-23,23])
plt.tight_layout()
plt.savefig(FIGDIR+name+".png")


############
# Breakout #
############

name = "Breakout"
with open(LOGDIR+'Breakout_s001.pkl', 'rb') as f:
    breakout_data = np.array(pickle.load(f))
    breakout_eps = np.array(pickle.load(f))
breakout_t         = (breakout_data[:,0]) / 1000000.0
breakout_mean      = breakout_data[:,1]
breakout_best_mean = breakout_data[:,2]
breakout_ep        = breakout_data[:,3]
breakout_eps_sm    = smoothed_block(breakout_eps, 100)

fig, axarr = plt.subplots(2,2, figsize=(15,12))
axarr[0,0].set_title(name+ " Scores at Timesteps", fontsize=title_size)
axarr[0,1].set_title(name+ " Scores at Timesteps (Block 100)", fontsize=title_size)
axarr[0,0].plot(breakout_t, breakout_ep, c='blue', lw=lw, label='Seed 0')
axarr[0,1].plot(breakout_t, breakout_mean, c='blue', lw=lw, label='Seed 0')
axarr[0,0].set_xlabel("Training Steps (in Millions)", fontsize=axis_size)
axarr[0,1].set_xlabel("Training Steps (in Millions)", fontsize=axis_size)

axarr[1,0].set_title(name+ " Scores per Episode", fontsize=title_size)
axarr[1,1].set_title(name+ " Scores per Episode (Block 100)", fontsize=title_size)
axarr[1,0].plot(breakout_eps, c='blue', lw=lw, label='Seed 0')
axarr[1,1].plot(breakout_eps_sm, c='blue', lw=lw, label='Seed 0')
axarr[1,0].set_xlabel("Number of Episodes", fontsize=axis_size)
axarr[1,1].set_xlabel("Number of Episodes", fontsize=axis_size)

for i in range(2):
    for j in range(2):
        axarr[i,j].set_ylabel("Rewards", fontsize=axis_size)
        axarr[i,j].tick_params(axis='x', labelsize=tick_size)
        axarr[i,j].tick_params(axis='y', labelsize=tick_size)
        axarr[i,j].legend(loc='upper left', prop={'size':legend_size})

plt.tight_layout()
plt.savefig(FIGDIR+name+".png")
