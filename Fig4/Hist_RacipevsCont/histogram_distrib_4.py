import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import jensenshannon
from matplotlib import rcParams
import time
from os import listdir
from os.path import isfile, join
import matplotlib
import histarrows as histarrows

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles4/") if isfile(join("topofiles4/", f))]


matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'40', 'axes.labelsize':'35', 'xtick.labelsize':'33', 'ytick.labelsize':'30', 'legend.fontsize':'30'})


jsdarr=[]
coords = []


for i in range(len(topofiles)):
    
    try:
         racprob=np.loadtxt("racdata/{}_racipe_probfull_processed.txt".format(topofiles[i]))
         racarr=racprob
    except:
         pass
        
   
    boolprob=np.loadtxt("contdata/{}_ising_probfull.txt".format(topofiles[i]))  ##this is actually cont data, not boolean/ising.
    boolarr=boolprob.T[1]
    
    jsd=jensenshannon(boolarr,racarr,2)
    jsdarr.append(jsd)
    if(topofiles[i][0]!='r'):
        print(topofiles[i], jsd)
        coords.append(jsd)

number = 4

n_bins  = 20  
r = 2
matplotlib.rcParams.update({'font.size': 13*r})  
fig,ax = plt.subplots()
valarr = jsdarr
names = ["GRHL2" , "GRHL2wa", "OVOL", "OVOLsi"]
colours = ['r', 'g', 'm', 'k']
histarrows.histogram(ax, valarr, coords, names, colours, n_bins)   

plt.xlabel("JSD b/w RACIPE and Cont.", fontweight="bold", c = '0.3')
plt.ylabel("No. of random networks" , fontweight="bold", c = '0.3')
#plt.title("Networks (Size 4)", fontweight="bold", c = '0.3' ,  fontsize = 30 ) 

       
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)     
plt.tight_layout()       
plt.savefig("racboolhist_size4.png", transparent = True)


  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    