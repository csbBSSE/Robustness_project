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
                xytext=(xcoord, ycoord + 4), textcoords='data',
                arrowprops=dict(arrowstyle="->",color = colours[i], lw = 3,
                                connectionstyle="arc3"),
                )
    legend_ele = []
    for i in range(num):
        legend_ele.append(Line2D([0], [0], marker='o', color='w', label=names[i], markerfacecolor=colours[i], markersize=20))
    ax.legend(handles = legend_ele, frameon= False,  handletextpad=0.01,)

'''
fig,ax = plt.subplots()
valarr = np.random.normal(0, 1, 100)
colours = ['r', 'g', 'b', 'k']
coords = [0.2, 0.3, 0.5, 0.7]
names = ["GRHL2", "GRHL2", "GRHL2", "GRHL2"]
nbins = 20
histogram(ax, valarr, coords, names, colours, nbins)
plt.savefig("histname.png")
'''