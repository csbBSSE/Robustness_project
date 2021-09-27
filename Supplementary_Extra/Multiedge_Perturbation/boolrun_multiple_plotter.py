import os
import time
from os import listdir
from os.path import isfile, join
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
import pandas as pd
import matplotlib

topofiles.sort()

import numpy as np
from scipy.spatial.distance import jensenshannon

import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '25', 'axes.titlesize':'25', 'axes.labelsize':'25', 'xtick.labelsize':'15', 'ytick.labelsize':'15', 'legend.fontsize':'20'})

print(topofiles)
####


num_sim = 10

tottime = time.time()


for i in range(len(topofiles)):
    curnetwork = open("curnetwork.txt", 'w')
    looptime = time.time()
    file = open("topofiles/{}.topo".format(topofiles[i]),"r").read().split("\n")
    edgecount=len(file)-1
    

    print(topofiles[i])

    stabprob=np.loadtxt("JSD/{}_jsd.txt".format(topofiles[i]))
 
    
    #print(cycles)
   
    
    jsdarr=np.zeros((edgecount,num_sim))
    
    jsdavg=np.zeros(edgecount)
    jsdstd=np.zeros(edgecount)
    
    for j in range(edgecount):
        for k in range(num_sim):
        
            jsdarr[j][k]=jensenshannon(stabprob[0],stabprob[1+j*num_sim+k] ,2)
            
        jsdavg[j]=np.mean(jsdarr[j])
        jsdstd[j]=np.std(jsdarr[j])
        
    xarr=[j for j in range(edgecount)]
    
    
    r = 2
    fig = plt.figure()
    matplotlib.rcParams.update({'font.size': 10*r})
    matplotlib.rcParams.update({'errorbar.capsize': 2*r})
    
    plt.scatter(xarr,jsdavg)
    plt.errorbar(xarr,jsdavg,yerr=jsdstd,ecolor='k' , linewidth = 3)
    
    plt.xlabel("No. of edges perturbed",fontweight="bold" , c='0.3')
    plt.ylabel("Avg. Perturbation JSD",fontweight="bold" , c='0.3')
    l = len(jsdavg)
    
    plt.axhline(y= jsdavg[1], color='r', linestyle='--' , linewidth= 3)
    plt.axhline(y= jsdavg[l-1], color='r', linestyle='--', linewidth= 3)
    plt.xticks(np.arange(min(xarr), max(xarr)+2, 2.0))
    plt.title("{} JSD Curve".format(topofiles[i]) ,fontweight="bold" , c='0.3')
    plt.tight_layout()
    plt.savefig("Figs/{}_jsdcurve_new.png".format(topofiles[i]) , transparent = True)
    plt.clf()
  