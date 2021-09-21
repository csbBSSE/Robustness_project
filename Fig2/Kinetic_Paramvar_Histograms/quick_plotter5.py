import os
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import modules.metric as metric
from scipy.stats import pearsonr
import sys
import histarrows as histarrows
sys.path.append('../')

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles5/") if isfile(join("topofiles5/", f))]
topofiles.sort()
matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '25', 'axes.titlesize':'25', 'axes.labelsize':'25', 'xtick.labelsize':'20', 'ytick.labelsize':'20', 'legend.fontsize':'20'})


plastarr =[]
jsdarr =[]

cyclearr=[]
cyclearr2=[]
jsdarr2 =[]

for i in topofiles:
    
    kineticjsd = open("kinetic_jsd.txt", 'r').readlines()
    kineticplast = open("kinetic_plast.txt", 'r').readlines()
    
    flag = 0
    plastval = 0
    jsdval = 0
    
    for j in kineticjsd:
        k = j.split(" ")
        if(k[0]==i):
            flag = 1
            jsdval = float(k[1])
            break
            
    for j in kineticplast:
        k = j.split(" ")
        if(k[0]==i):
            flag = 1
            plastval = float(k[1])
            break    
    if(flag ==0):
        continue
    
    plastarr.append(plastval)
    jsdarr.append(jsdval)
    
    network_name = i
    graph = metric.networkx_graph(network_name)
    cycle_stuff = metric.cycle_info(network_name, graph)
    
    cyclearr.append(cycle_stuff[6])

#-----------------------------------------------------------------------------------------------------------------
number = 5

n_bins  = 20  
r = 2
matplotlib.rcParams.update({'font.size': 13*r})  
fig,ax = plt.subplots()
valarr = plastarr
coords = [0.836]
names = ["OCT4"]
colours = ['r']
histarrows.histogram(ax, valarr, coords, names, colours, n_bins)   


plt.xlabel("Avg. Fold Change in Plasticity", fontweight="bold", c = '0.3')
plt.ylabel("Number of Random Networks", fontweight="bold", c = '0.3')
plt.title("Avg. Fold Change in Plasticity (Kinetic) : Size {}".format(number), fontweight="bold", c = '0.3' , size = 13*r)

f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)    
plt.tight_layout()         
plt.savefig("kineticplastavg{}hist.jpg".format(number), transparent = True)



#-----------------------------------------------------------------------------------------------------------------
number = 5

n_bins  = 20  
r = 2
matplotlib.rcParams.update({'font.size': 13*r})  
fig,ax = plt.subplots()
valarr = jsdarr
coords = [0.0647804]
names = ["OCT4"]
colours = ['r']
histarrows.histogram(ax, valarr, coords, names, colours, n_bins)   

plt.xlabel("Avg. Parameter Variation JSD", fontweight="bold", c = '0.3')
plt.ylabel("Number of Random Networks", fontweight="bold", c = '0.3')
plt.title("Avg. Parameter Variation JSD Distribution : Size {}".format(number), fontweight="bold", c = '0.3' , size = 13*r)

f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)    
plt.tight_layout()        
plt.savefig("kineticjsdavg{}hist.jpg".format(number), transparent = True)







