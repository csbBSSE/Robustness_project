import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import pow
import os
from scipy.stats import norm, zscore
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from matplotlib import rcParams
import pandas as pd
import os
import time
from os import listdir
from os.path import isfile, join



topofiles= [os.path.splitext(f)[0] for f in listdir("topofilesisingcnt/") if isfile(join("topofilesisingcnt/", f))]
jsdising =[]
jsdcnt2 =[]
neg_weight_fracarr = []
cycledata=open("networkCycles.txt").read()
cycledata=cycledata.split("\n")

for w in range(len(topofiles)):
    #arr1=[0,0.0224,0,0.4718,0.4793,0,0.0265,0]

    #probjsd1 = np.loadtxt("GRHL2_async_unweigh_jsd_{}.txt".format(1))
    #arr1= probjsd1[99,:]
    racipe = 0
    try:
        racipe = np.loadtxt("JSD_racipe/{}_racipe.txt".format(topofiles[w]))
    except:
        continue
    for k in range(len(cycledata)):
          data1=cycledata[k].split()
                #print(data1)
          if(data1[0]==topofiles[w]):       
            negweightfrac_val = 1- float(data1[3])
            neg_weight_fracarr.append(negweightfrac_val)
            break
        
    ising = np.loadtxt("JSD_ising/{}_JSD.txt".format(topofiles[w]))
    cont = np.loadtxt("JSD_cnt2/{}_JSD.txt".format(topofiles[w]))

    jsd1 = jensenshannon(ising,racipe,2)
    jsd2 = jensenshannon(cont,racipe,2)
    jsdising.append(jsd1)
    jsdcnt2.append(jsd2)
    
fig, ax = plt.subplots(ncols=1)
r = 2
matplotlib.rcParams.update({'font.size': 10*r})

sc = ax.scatter(jsdcnt2, jsdising, c= neg_weight_fracarr)
plt.xlabel("RACIPE JSD from Cont.", fontweight = 'bold' , c='0.3' , size = 15*r)
plt.ylabel("RACIPE JSD from Boolean", fontweight = 'bold' , c='0.3',size = 15*r)
#plt.title("Boolean JSD vs Cont. JSD" , fontweight = 'bold' , c='0.3',size = 15*r)
plt.xticks(fontsize = 10*r)
plt.yticks(fontsize = 10*r)
x = np.linspace(*ax.get_xlim())
ax.plot(x, x , c='r')

plt.colorbar(sc)
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)


plt.savefig("compare_bool_cont.png",transparent = True)

plt.clf()

fig, ax = plt.subplots(ncols=1)
r = 2
matplotlib.rcParams.update({'font.size': 10*r})

sc = ax.scatter(jsdcnt2, jsdising)
plt.xlabel("RACIPE JSD from Cont.", fontweight = 'bold' , c='0.3' , size = 15*r)
plt.ylabel("RACIPE JSD from Boolean", fontweight = 'bold' , c='0.3',size = 15*r)
#plt.title("Boolean JSD vs Cont. JSD" , fontweight = 'bold' , c='0.3',size = 10*r)
plt.xticks(fontsize = 10*r)
plt.yticks(fontsize = 10*r)
x = np.linspace(*ax.get_xlim())
ax.plot(x, x , c='r' , linewidth = 3)

f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)


plt.savefig("compare_bool_cont_nocolor.png",transparent = True)

plt.clf()




