import numpy as np
import matplotlib.pyplot as plt
from math import pow
import os
from scipy.stats import norm, zscore
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from matplotlib import rcParams
import pandas as pd
from scipy.stats import pearsonr
import os
import time
from os import listdir
from os.path import isfile, join
import matplotlib
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles5/") if isfile(join("topofiles5/", f))]
import histarrows as histarrows


matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'40', 'axes.labelsize':'35', 'xtick.labelsize':'33', 'ytick.labelsize':'30', 'legend.fontsize':'30'})


jsdarr=[]
coords = []
racarr = []
for i in range(len(topofiles)):
    
    try:
         racprob=np.loadtxt("racdata/{}_racipe_probfull_processed.txt".format(topofiles[i]))
         racarr=racprob
    except:
         continue
         pass
    
    boolprob=np.loadtxt("booldata/{}_ising_probfull.txt".format(topofiles[i]))
    boolarr=boolprob.T[1]
    jsd=jensenshannon(boolarr,racarr,2)
    jsdarr.append(jsd)
    if(topofiles[i][0]!='r'):
        print(topofiles[i], jsd)
        coords.append(jsd)

number = 5

n_bins  = 20  
r = 2
matplotlib.rcParams.update({'font.size': 13*r})  
fig,ax = plt.subplots()
valarr = jsdarr
valarr.append(0.1396)
names = ["OCT4"]
colours = ['r']
coords = [0.1396]
histarrows.histogram(ax, valarr, coords, names, colours, n_bins)   

plt.xlabel("JSD b/w RACIPE and Cont.", fontweight="bold", c = '0.3')
plt.ylabel("No. of random networks" , fontweight="bold", c = '0.3')
#plt.title("Networks (Size 5)", fontweight="bold", c = '0.3' , fontsize = 29)
plt.yticks(np.arange(0,22,3))
       
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)     
plt.tight_layout()       
plt.savefig("racboolhist_size5.png", transparent = True)


    
    
    
    
    
    
    
    
    
    
    
    
    