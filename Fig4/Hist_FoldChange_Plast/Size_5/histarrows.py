import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

shapedict =	{
  "GRHL2": "D",
  "GRHL2wa": "X",
  "OVOL": "^",
  "OVOLsi": "s",
  "OCT4": "*",
  "NRF2": "P"
}

shapesizedict =	{
  "D": 20,
  "X": 23,
  "^": 22,
  "s": 20,
  "*": 29,
  "P": 25
}


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def histogram(ax, valarr, coords, names, colours, nbins, error):
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
    net_names = names[:]    
    for i in range(num):
        names[i] = names[i] + " [{}±{}]".format(error[i][0], error[i][1])
    for i in range(num):
        legend_ele.append(Line2D([0], [0], marker=shapedict[net_names[i]], color='w', label=names[i], markerfacecolor=colours[i],markersize=shapesizedict[shapedict[net_names[i]]],linestyle = 'None'))
    ax.legend(handles = legend_ele, frameon= False,  handletextpad=0.01 , loc='upper left')

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