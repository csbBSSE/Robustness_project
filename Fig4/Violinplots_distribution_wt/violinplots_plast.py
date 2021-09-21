import matplotlib
import os
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import seaborn 


min_fold = []
max_fold = []
mean_fold = []
grhl2_fold = []


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


data = np.loadtxt("GRHL2_plastdata.txt")
for a in data[1:]:
    foldchange = 0
    wt = data[0]

    if (wt == 0 and a != 0) or (wt != 0 and a == 0):
        foldchange = 0
    elif (wt == 0 and a == 0):
        foldchange = 1
    else:
        foldchange = min(wt / a, a / wt)
    grhl2_fold.append(foldchange)


r = 2

matplotlib.rcParams.update({'font.size': 10*r, 'text.color': '0.3', 'axes.labelweight':'bold', 'font.weight': 'bold', 'axes.labelcolor': '0.3', 'axes.titlecolor': '0.3',  'ytick.labelsize': 7*r ,'xtick.color':'0.3', 'ytick.color':'0.3'})

fig, ax = plt.subplots()
ax.set_aspect(6)
fig.set_size_inches((2.5, 5))

data_matrix = [grhl2_fold]

labels = ['GRHL2']

ax = seaborn.violinplot( data = data_matrix ,inner=None , bw = 0.5, cut=0 ,palette=['r'])
ax.set_xticklabels(labels)
a1 = np.mean(data_matrix[0])
plt.scatter([0], [a1] , c='k' )
plt.yticks(np.arange(0.3,1,0.2))

ax.set_ylabel("Fold Change")
plt.tight_layout()

plt.savefig("violinplot_plast_wt.png")
