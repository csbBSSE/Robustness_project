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

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'40', 'axes.labelsize':'35', 'xtick.labelsize':'33', 'ytick.labelsize':'30', 'legend.fontsize':'30'})
plt.rcParams['figure.dpi'] = 500

number = 8
n_bins = 20

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles8/") if isfile(join("topofiles8/", f))]
topofiles.sort()
def rankcalc(coords, valarr, names):
    n = len(valarr)
    for i in range(len(coords)):
        perc = 0
        for j in range(len(valarr)):
            if(valarr[j]>coords[i]):
                perc+=1
        perc = (np.round((perc/n)*1000))/1000
        print(names[i], perc)
avgarr=[]
stdarr=[]
coords = []
for i in topofiles:
    a=np.loadtxt("raw_data/cnt2_{}_jsd.txt".format(i))
    b=np.mean(a)
    avgarr.append(b)
    if(i[0]!='r'):
        coords.append(b)
        
number = 8
n_bins  = 20  
r = 2
matplotlib.rcParams.update({'font.size': 10*r})  
fig,ax = plt.subplots()
valarr = avgarr
names = ["NRF2"]
colours = ['r']
error = [[0.23	,0.03]]

histarrows.histogram(ax, valarr, coords, names, colours, n_bins, error)   

plt.xlabel("Avg. Perturbation JSD", fontweight="bold", c = '0.3')
plt.ylabel("No. of Random Networks", fontweight="bold", c = '0.3')
#plt.title("Networks (Size {})".format(number), fontweight="bold", c = '0.3', fontsize = 30)

f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)  
plt.ylim([0,19])       
plt.tight_layout()

plt.savefig("jsdavg{}hist.png".format(number), transparent = True)
plt.clf()
rankcalc(coords, valarr, names)




    