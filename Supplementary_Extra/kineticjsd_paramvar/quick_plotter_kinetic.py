from os.path import splitext, isfile, join
from os import listdir
import os
import networkx as nx
import numpy as np
import modules.metric as metric
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import sys
sys.path.append('../')

print('imported modules')
def func(X, a , b , c):
    x,y = X
    z = np.zeros(len(x))
    z1 = [c]*len(x)
    z1 = np.array(z1)
    for i in range(len(x)):
        if(x[i] == 0 and y[i] == 0):
          z[i] = a*x[i] / max(a*x[i] + y[i] , 1)
        else:
            z[i] = a*x[i]/(a*x[i] + y[i])
    return b*z + z1

def func_output(X, a , b , c):
    x,y = X
    z = np.zeros(len(x))
    z1 = [c]*len(x)
    z1 = np.array(z1)
    for i in range(len(x)):
        if(x[i] == 0 and y[i] == 0):
          z[i] = a*x[i] / max(a*x[i] + y[i] , 1)
        else:
            z[i] = a*x[i]/(a*x[i] + y[i])
    return z


topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()


jsdarr =[]
weightpos = []
weightneg = []

for i in topofiles:
    kineticjsd = open("kinetic_jsd.txt", 'r').readlines()

    flag = 0
    plastval = 0
    jsdval = 0
    
    for j in kineticjsd:
        k = j.split(" ")
        if(k[0]==i):
            flag = 1
            jsdval = float(k[1])
            break
            
    if(flag ==0):
        continue
    
    jsdarr.append(jsdval)
    
    network_name = i
    graph = metric.networkx_graph(network_name)
    cycle_stuff = metric.cycle_info(network_name, graph)
    weightpos.append(cycle_stuff[4])
    weightneg.append(cycle_stuff[5])

r = 2
fig = plt.figure()
matplotlib.rcParams.update({'font.size': 10*r})
    


x_arr1 = weightpos
x_arr2 = weightneg
y_arr = jsdarr
p0 = 1.0 , 1.0 , 1.0
popt, pcov = curve_fit(func, (x_arr1,x_arr2), y_arr, p0)
z_arr = func_output( (x_arr1,x_arr2) , popt[0], popt[1] , popt[2])
x_arr = np.array(z_arr)   
print(popt)

plt.scatter(x_arr,jsdarr)
corr, _ = pearsonr(x_arr,jsdarr)
plt.xlabel("Fraction of Weighted Positive Cycles", fontweight = 'bold' , c='0.3')
plt.ylabel("Avg. Kinetic JSD", fontweight = 'bold' , c='0.3')
plt.title("Random Networks (All Sizes):        ρ = {:.3f}".format(corr), fontweight = 'bold' , c='0.3')


f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)
plt.savefig("kineticjsd.jpg")
plt.clf()    
    


