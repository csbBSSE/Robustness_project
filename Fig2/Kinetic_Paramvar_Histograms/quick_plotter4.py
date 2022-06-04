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

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles4/") if isfile(join("topofiles4/", f))]
topofiles.sort()

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'40', 'axes.labelsize':'35', 'xtick.labelsize':'30', 'ytick.labelsize':'30', 'legend.fontsize':'28'})

def rankcalc(coords, valarr, names):
    n = len(valarr)
    for i in range(len(coords)):
        perc = 0
        for j in range(len(valarr)):
            if(valarr[j]>coords[i]):
                perc+=1
        perc = (np.round((perc/n)*1000))/1000
        print(names[i], perc)
        
fsize = 20
plastarr =[]
jsdarr =[]

cyclearr=[]
cyclearr2=[]
jsdarr2 =[]
kineticjsd = open("kinetic_jsd.txt", 'r').readlines()
kineticplast = open("kinetic_plast.txt", 'r').readlines()


for i in topofiles:
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
   

#-----------------------------------------------------------------------------------------------------------------
number = 4

n_bins  = 20  
r = 2
matplotlib.rcParams.update({'font.size': 13*r})  
fig,ax = plt.subplots()
valarr = plastarr
coords = [0.853, 0.873, 0.855, 0.846]
names = ["GRHL2" , "GRHL2wa", "OVOL", "OVOLsi"]
colours = ['r', 'g', 'm', 'k']
error = [[0.84,	0.04]	,[0.74,	0.03],	[0.93,	0.03],	[0.89,	0.03]]
fsize = 18

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'40', 'axes.labelsize':'35', 'xtick.labelsize':'30', 'ytick.labelsize':'30', 'legend.fontsize':'27'})



histarrows.histogram(ax, valarr, coords, names, colours, n_bins,error ,fsize)   



plt.xlabel("Avg. Fold Change in Plasticity\n(Dynamic)", fontweight="bold", c = '0.3')
plt.ylabel("No. of Random Networks", fontweight="bold", c = '0.3')
plt.title("Size {}".format(number), fontweight="bold", c = '0.3')
plt.yticks(np.arange(0,19,3))
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)    
plt.tight_layout()         
plt.savefig("kineticplastavg{}hist.png".format(number), transparent = True)
rankcalc(coords, valarr, names)

#-----------------------------------------------------------------------------------------------------------------
matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'40', 'axes.labelsize':'35', 'xtick.labelsize':'30', 'ytick.labelsize':'30', 'legend.fontsize':'28'})

number = 4
fsize = 20
n_bins  = 20  
r = 2
matplotlib.rcParams.update({'font.size': 13*r})  
fig,ax = plt.subplots()
valarr = jsdarr
coords = [0.062, 0.081, 0.046, 0.0554]
names = ["GRHL2" , "GRHL2wa", "OVOL", "OVOLsi"]
colours = ['r', 'g', 'm', 'k']
error = [[.17	,0.06],	[0.12	,0.06]	,[0.16,	0.08]	,[0.19,	0.06]]

histarrows.histogram(ax, valarr, coords, names, colours, n_bins ,error,fsize)   

plt.xlabel("Avg. Parameter Variation JSD", fontweight="bold", c = '0.3')
plt.ylabel("No. of Random Networks", fontweight="bold", c = '0.3')
plt.title("Size {}".format(number), fontweight="bold", c = '0.3')

f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)    
plt.tight_layout()        
plt.savefig("kineticjsdavg{}hist.png".format(number), transparent = True)
rankcalc(coords, valarr, names)





