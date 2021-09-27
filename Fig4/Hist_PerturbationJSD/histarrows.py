import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def histogram(ax, valarr, coords, names, colours, nbins):
    plt.hist(valarr, bins = nbins)
    height, tempvals = np.histogram(valarr, bins = nbins)
    
    vals = []
    for i in range(len(tempvals) - 1):
        vals.append(tempvals[i] + tempvals[i+1])
    vals = np.array(vals)
    vals /= 2

    num = len(coords)
    for i in range(num):
        ind = find_nearest(coords[i], vals)
        xcoord = coords[i]
        ycoord = height[ind]
        ax.annotate("",
                xy=(xcoord ,ycoord), xycoords='data',
                xytext=(xcoord, ycoord + 3), textcoords='data',
                arrowprops=dict(arrowstyle="->",color = colours[i], lw = 3,
                                connectionstyle="arc3"),
                )
    legend_ele = []
    for i in range(num):
        legend_ele.append(Line2D([0], [0], marker='o', color='w', label=names[i], markerfacecolor=colours[i], markersize=20))

    ax.legend(handles = legend_ele, bbox_to_anchor = (0.47,0.63), handletextpad=0.01, frameon=False)

