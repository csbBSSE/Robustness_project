import os
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '25', 'axes.titlesize':'25', 'axes.labelsize':'25', 'xtick.labelsize':'20', 'ytick.labelsize':'20', 'legend.fontsize':'20'})

topofiles = ["GRHL2", "NRF2", "GRHL2wa", "OCT4", "OVOL2", "OVOLsi"]
topofiles = ["GRHL2", "OVOL2", "OCT4"]

colours = ['r', 'g', 'b', 'k', 'c', 'm', 'y']
colcnt = 0
fig, ax = plt.subplots()

for i in topofiles:
    bool = np.loadtxt("{}_percerror_bool.txt".format(i))[2:]
    rac = np.loadtxt("{}_percerror_rac.txt".format(i))
    print(bool[:, 0])
    ax.plot(bool[:, 0]/100, bool[:, 1], 'o:', c = colours[colcnt])
    ax.plot(rac[:, 0], rac[:, 1], 'o-', c = colours[colcnt])
    colcnt += 1



f=(2)*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)
#plt.yticks([1, 0.1, 0.01, 0.001])
ax.set_xscale('log')
ax.set_yscale('log')
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
ax.set_xlabel("Number of RACIPE models")
ax.set_ylabel("Fraction Error of Phenotypic Distribution")
ax.set_title("RACIPE vs Continuous Phenotypic Error")

legend_ele = [Line2D([0], [0], marker='o', color='w', label='GRHL2', markerfacecolor='r', markersize=15),
              Line2D([0], [0], marker='o', color='w', label='OVOL2', markerfacecolor='g', markersize=15),
              Line2D([0], [0], marker='o', color='w', label='OCT4', markerfacecolor='b', markersize=15),
              Line2D([0], [0], linestyle='-', color='k', label='RACIPE', markerfacecolor='g', markersize=15),
              Line2D([0], [0], linestyle=':', color='k', label='Continuous', markerfacecolor='g', markersize=15)]
ax.legend(handles = legend_ele)

fig.tight_layout()
plt.savefig("errorplot.png")