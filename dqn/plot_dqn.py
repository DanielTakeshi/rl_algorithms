import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
plt.style.use('seaborn-darkgrid')
np.set_printoptions(edgeitems=100,
                    linewidth=100,
                    suppress=True)

# Some default settings.
LOGDIR = 'logs/'
FIGDIR = 'figures/'
title_size = 22
axis_size = 20
tick_size = 18
#xaxis_ticks = [0, 2000000, 4000000, 6000000, 8000000]
lw = 3


########
# Pong #
########

# Change all '-inf's to be -21.
pong_data = np.array( pickle.load(open(LOGDIR+'pong_s000.pkl', 'rb')) )
pong_data = np.maximum(pong_data, -21)
pong_t         = pong_data[:,0]
pong_mean      = pong_data[:,1]
pong_best_mean = pong_data[:,2]
pong_ep        = pong_data[:,3]

fig, axarr = plt.subplots(1,2, figsize=(15,6))
axarr[0].set_title("Pong Performance (Per Episode)", fontsize=title_size)
axarr[0].plot(pong_t, pong_ep, lw=lw)
axarr[1].set_title("Pong Performance (Blocks of 100)", fontsize=title_size)
axarr[1].plot(pong_t, pong_mean, lw=lw)

for i in range(2):
    axarr[i].set_xlabel("Episodes", fontsize=axis_size)
    axarr[i].set_ylabel("Rewards", fontsize=axis_size)
    axarr[i].set_xlim([0,max(pong_t)])
    axarr[i].set_ylim([-22,22])
    #axarr[i].set_xticks(xaxis_ticks)
    #axarr[i].ticklabel_format(style='plain')
    axarr[i].tick_params(axis='x', labelsize=tick_size)
    axarr[i].tick_params(axis='y', labelsize=tick_size)

plt.tight_layout()
plt.savefig(FIGDIR+"pong.png")
