import matplotlib
import os
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import seaborn 

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'35', 'axes.labelsize':'28', 'xtick.labelsize':'28', 'ytick.labelsize':'18', 'legend.fontsize':'30'})
plt.rcParams['figure.dpi'] = 500

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


grhl2_jsd = np.loadtxt("cont_GRHL2_jsd.txt")   


r = 2


fig, ax = plt.subplots()
ax.set_aspect(6)
fig.set_size_inches((3, 5))

data_matrix = [grhl2_jsd]

labels = ['GRHL2']

ax = seaborn.violinplot( data = data_matrix ,inner=None , bw = 0.5, cut=0 ,palette=['r'])
ax.set_xticklabels(labels)
a1 = np.mean(data_matrix[0])
plt.scatter([0], [a1] , c='k' )
plt.yticks(np.arange(0,1,0.2))
#plt.title("Distribution of Avg. Perturbation JSD (GRHL2)" , c= '0.3' , fontweight = 'bold', fontsize = 25)
ax.set_ylabel("Perturbation JSD")
plt.tight_layout()

plt.savefig("violinplot_jsd_wt.png", transparent = True)


