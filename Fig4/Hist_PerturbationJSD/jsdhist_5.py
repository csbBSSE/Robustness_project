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

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '25', 'axes.titlesize':'25', 'axes.labelsize':'25', 'xtick.labelsize':'20', 'ytick.labelsize':'20', 'legend.fontsize':'20'})


number = 5
n_bins = 20

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles5/") if isfile(join("topofiles5/", f))]
topofiles.sort()
avgarr=[]
stdarr=[]

for i in topofiles:
    a=np.loadtxt("raw_data/cnt2_{}_jsd.txt".format(i))
    b=np.mean(a)
    avgarr.append(b)

number = 5
n_bins  = 20  
r = 2
matplotlib.rcParams.update({'font.size': 10*r})  
fig,ax = plt.subplots()
valarr = avgarr
names = ["OCT4"]
colours = ['r']
coords = [0.2]   
histarrows.histogram(ax, valarr, coords, names, colours, n_bins)   

plt.xlabel("Average Perturbation JSD", fontweight="bold", c = '0.3', fontsize = 30)
plt.ylabel("Number of Random Networks", fontweight="bold", c = '0.3', fontsize = 30)
plt.title("Perturbation JSD Distribution: Size {}".format(number), fontweight="bold", c = '0.3', fontsize = 30)

f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)            
plt.tight_layout()

plt.savefig("jsdavg{}hist.jpg".format(number), transparent = True)
plt.clf()





    