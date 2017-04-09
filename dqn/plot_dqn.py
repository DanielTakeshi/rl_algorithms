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
legend_size = 20
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
fig, axarr = plt.subplots(2,2, figsize=(15,12))

pong_data = []
pong_eps = []
pong_t = []
pong_mean = []
pong_best_mean = []
pong_ep = []
pong_eps_sm = []
pong_colors = ['red','black']
pong_labels = ['seed=1','seed=2']

for i in range(0,2):
    index_str = str(i+1)
    with open(LOGDIR+'Pong_s00'+index_str+'.pkl', 'rb') as f:
        pong_data.append( np.array(pickle.load(f)) )
        pong_eps.append( np.array(pickle.load(f)) )
    pong_data[i] = np.maximum(pong_data[i], -21)
    pong_t.append((pong_data[i][:,0]) / 1000000.0)
    pong_mean.append(pong_data[i][:,1])
    pong_best_mean.append(pong_data[i][:,2])
    pong_ep.append(pong_data[i][:,3])
    pong_eps_sm.append(smoothed_block(pong_eps[i], 100))

    axarr[0,0].set_title(name+ " Scores at Timesteps", fontsize=title_size)
    axarr[0,1].set_title(name+ " Scores at Timesteps (Block 100)", fontsize=title_size)
    axarr[0,0].plot(pong_t[i], pong_ep[i], c=pong_colors[i], lw=lw,
                    label=pong_labels[i])
    axarr[0,1].plot(pong_t[i], pong_mean[i], c=pong_colors[i], lw=lw, 
                    label=pong_labels[i])
    axarr[0,0].set_xlabel("Training Steps (in Millions)", fontsize=axis_size)
    axarr[0,1].set_xlabel("Training Steps (in Millions)", fontsize=axis_size)
    
    axarr[1,0].set_title(name+ " Scores per Episode", fontsize=title_size)
    axarr[1,1].set_title(name+ " Scores per Episode (Block 100)", fontsize=title_size)
    axarr[1,0].plot(pong_eps[i], c=pong_colors[i], lw=lw, 
                    label=pong_labels[i])
    axarr[1,1].plot(pong_eps_sm[i], c=pong_colors[i], lw=lw, 
                    label=pong_labels[i])
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
fig, axarr = plt.subplots(2,2, figsize=(15,12))

breakout_data = []
breakout_eps = []
breakout_t = []
breakout_mean = []
breakout_best_mean = []
breakout_ep = []
breakout_eps_sm = []
breakout_colors = ['blue','yellow']
breakout_labels = ['seed=1','seed=2']

for i in range(0,2):
    index_str = str(i+1)
    with open(LOGDIR+'Breakout_s00'+index_str+'.pkl', 'rb') as f:
        breakout_data.append( np.array(pickle.load(f)) )
        breakout_eps.append( np.array(pickle.load(f)) )
    breakout_data[i] = np.maximum(breakout_data[i], -21)
    breakout_t.append((breakout_data[i][:,0]) / 1000000.0)
    breakout_mean.append(breakout_data[i][:,1])
    breakout_best_mean.append(breakout_data[i][:,2])
    breakout_ep.append(breakout_data[i][:,3])
    breakout_eps_sm.append(smoothed_block(breakout_eps[i], 100))

    axarr[0,0].set_title(name+ " Scores at Timesteps", fontsize=title_size)
    axarr[0,1].set_title(name+ " Scores at Timesteps (Block 100)", fontsize=title_size)
    axarr[0,0].plot(breakout_t[i], breakout_ep[i], c=breakout_colors[i], lw=lw,
                    label=breakout_labels[i])
    axarr[0,1].plot(breakout_t[i], breakout_mean[i], c=breakout_colors[i], lw=lw, 
                    label=breakout_labels[i])
    axarr[0,0].set_xlabel("Training Steps (in Millions)", fontsize=axis_size)
    axarr[0,1].set_xlabel("Training Steps (in Millions)", fontsize=axis_size)
    
    axarr[1,0].set_title(name+ " Scores per Episode", fontsize=title_size)
    axarr[1,1].set_title(name+ " Scores per Episode (Block 100)", fontsize=title_size)
    axarr[1,0].plot(breakout_eps[i], c=breakout_colors[i], lw=lw, 
                    label=breakout_labels[i])
    axarr[1,1].plot(breakout_eps_sm[i], c=breakout_colors[i], lw=lw, 
                    label=breakout_labels[i])
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


#############
# BeamRider #
#############

name = "BeamRider"
fig, axarr = plt.subplots(2,2, figsize=(15,12))

beamrider_data = []
beamrider_eps = []
beamrider_t = []
beamrider_mean = []
beamrider_best_mean = []
beamrider_ep = []
beamrider_eps_sm = []
beamrider_colors = ['orange','purple']
beamrider_labels = ['seed=1','seed=2']

for i in range(0,2):
    index_str = str(i+1)
    with open(LOGDIR+'BeamRider_s00'+index_str+'.pkl', 'rb') as f:
        beamrider_data.append( np.array(pickle.load(f)) )
        beamrider_eps.append( np.array(pickle.load(f)) )
    beamrider_data[i] = np.maximum(beamrider_data[i], -21)
    beamrider_t.append((beamrider_data[i][:,0]) / 1000000.0)
    beamrider_mean.append(beamrider_data[i][:,1])
    beamrider_best_mean.append(beamrider_data[i][:,2])
    beamrider_ep.append(beamrider_data[i][:,3])
    beamrider_eps_sm.append(smoothed_block(beamrider_eps[i], 100))

    axarr[0,0].set_title(name+ " Scores at Timesteps", fontsize=title_size)
    axarr[0,1].set_title(name+ " Scores at Timesteps (Block 100)", fontsize=title_size)
    axarr[0,0].plot(beamrider_t[i], beamrider_ep[i], c=beamrider_colors[i], lw=lw,
                    label=beamrider_labels[i])
    axarr[0,1].plot(beamrider_t[i], beamrider_mean[i], c=beamrider_colors[i], lw=lw, 
                    label=beamrider_labels[i])
    axarr[0,0].set_xlabel("Training Steps (in Millions)", fontsize=axis_size)
    axarr[0,1].set_xlabel("Training Steps (in Millions)", fontsize=axis_size)
    
    axarr[1,0].set_title(name+ " Scores per Episode", fontsize=title_size)
    axarr[1,1].set_title(name+ " Scores per Episode (Block 100)", fontsize=title_size)
    axarr[1,0].plot(beamrider_eps[i], c=beamrider_colors[i], lw=lw, 
                    label=beamrider_labels[i])
    axarr[1,1].plot(beamrider_eps_sm[i], c=beamrider_colors[i], lw=lw, 
                    label=beamrider_labels[i])
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



##########
# Enduro #
##########

name = "Enduro"
fig, axarr = plt.subplots(2,2, figsize=(15,12))

enduro_data = []
enduro_eps = []
enduro_t = []
enduro_mean = []
enduro_best_mean = []
enduro_ep = []
enduro_eps_sm = []
enduro_colors = ['gold','midnightblue']
enduro_labels = ['seed=1','seed=2']

for i in range(0,2):
    index_str = str(i+1)
    with open(LOGDIR+'Enduro_s00'+index_str+'.pkl', 'rb') as f:
        enduro_data.append( np.array(pickle.load(f)) )
        enduro_eps.append( np.array(pickle.load(f)) )
    enduro_data[i] = np.maximum(enduro_data[i], -21)
    enduro_t.append((enduro_data[i][:,0]) / 1000000.0)
    enduro_mean.append(enduro_data[i][:,1])
    enduro_best_mean.append(enduro_data[i][:,2])
    enduro_ep.append(enduro_data[i][:,3])
    enduro_eps_sm.append(smoothed_block(enduro_eps[i], 100))

    axarr[0,0].set_title(name+ " Scores at Timesteps", fontsize=title_size)
    axarr[0,1].set_title(name+ " Scores at Timesteps (Block 100)", fontsize=title_size)
    axarr[0,0].plot(enduro_t[i], enduro_ep[i], c=enduro_colors[i], lw=lw,
                    label=enduro_labels[i])
    axarr[0,1].plot(enduro_t[i], enduro_mean[i], c=enduro_colors[i], lw=lw, 
                    label=enduro_labels[i])
    axarr[0,0].set_xlabel("Training Steps (in Millions)", fontsize=axis_size)
    axarr[0,1].set_xlabel("Training Steps (in Millions)", fontsize=axis_size)
    
    axarr[1,0].set_title(name+ " Scores per Episode", fontsize=title_size)
    axarr[1,1].set_title(name+ " Scores per Episode (Block 100)", fontsize=title_size)
    axarr[1,0].plot(enduro_eps[i], c=enduro_colors[i], lw=lw, 
                    label=enduro_labels[i])
    axarr[1,1].plot(enduro_eps_sm[i], c=enduro_colors[i], lw=lw, 
                    label=enduro_labels[i])
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
