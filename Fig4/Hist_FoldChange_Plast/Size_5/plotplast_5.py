import os
import time
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import histarrows as histarrows
n_bins = 20

number = 5

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles5/") if isfile(join("topofiles5/", f))]

topofiles.sort()

print(topofiles)
def rankcalc(coords, valarr, names):
    n = len(valarr)
    for i in range(len(coords)):
        perc = 0
        for j in range(len(valarr)):
            if(valarr[j]>coords[i]):
                perc+=1
        perc = (np.round((perc/n)*1000))/1000
        print(names[i], perc)
plast=np.loadtxt("plastnetwork.txt")

#---------------------------------------------


matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'40', 'axes.labelsize':'35', 'xtick.labelsize':'33', 'ytick.labelsize':'30', 'legend.fontsize':'30'})
number = 5
n_bins  = 20  
r = 2
matplotlib.rcParams.update({'font.size': 10*r})  
fig,ax = plt.subplots()
valarr = plast
names = ["OCT4"]
colours = ['r']
coords = [0.86]
error = [[0.22,0.06]]
histarrows.histogram(ax, valarr, coords, names, colours, n_bins, error)   

plt.xlabel("Avg. fold change in Plasticity", fontweight="bold" , c='0.3')
plt.ylabel("No. of Random networks" , fontweight="bold" , c='0.3')
#plt.title("Distribution of Average fold change in Plasticity: Size {}".format(number), fontweight="bold" , c='0.3')

    
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)
ax.set_ylim([0,42])
plt.tight_layout()
print("i")
plt.savefig("plastfoldhist5.jpg", transparent = True)


plt.clf()
rankcalc(coords, valarr, names)
#---------------------------------------------




    
