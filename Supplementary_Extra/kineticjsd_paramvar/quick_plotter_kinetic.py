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
from matplotlib.lines import Line2D
import sys
sys.path.append('../')

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'30', 'axes.labelsize':'30', 'xtick.labelsize':'25', 'ytick.labelsize':'25', 'legend.fontsize':'25'})


colorarr = []
col = ['r' , 'g', 'm', 'k' ,'c', 'y', 'C1' ]

print('imported modules')
def starfunc(significance):
    if significance < 0.001:
        return "***"
    elif significance < 0.01:
        return "**"
    elif significance < 0.05:
        return "*"
    else:
        return ""
        
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
    kineticjsd = open("raw_data/kinetic_jsd.txt", 'r').readlines()

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

fig,ax = plt.subplots()


x_arr1 = weightpos
x_arr2 = weightneg
y_arr = jsdarr
p0 = 1.0 , 1.0 , 1.0
popt, pcov = curve_fit(func, (x_arr1,x_arr2), y_arr, p0)
z_arr = func_output( (x_arr1,x_arr2) , popt[0], popt[1] , popt[2])
x_arr = np.array(z_arr)   



plt.scatter(x_arr,jsdarr)
sns.regplot(x_arr,jsdarr,line_kws = {"color": 'b'} )
ax.lines.pop(0)
pcorr, significance = pearsonr(x_arr,jsdarr)


colarr = []
count = 0
x_arr1 = []
y_arr1 = []
labels = []
test = []
test.append(Line2D([], [], color= 'w', marker='X', markersize=20, label= "ρ = {:.3f}{}".format(pcorr, starfunc(significance)) ,linestyle='None'))

y_arr = jsdarr

while(topofiles[count][0]!='r'):
           labels.append(topofiles[count])
           colarr.append(col[count])
           x_arr1.append(x_arr[count])
           y_arr1.append(y_arr[count])
           #print("labels",labels)
           test.append(Line2D([], [], color= col[count], marker='X',
                          markersize=20, label= labels[count],linestyle='None'))
           count+=1
for k in range(len(x_arr1)):
            plt.plot( x_arr1[k], y_arr1[k] , c =  colarr[k] , marker = 'X',  markersize = 20 , linestyle = 'None')    
            
            
print(popt)

plt.xlabel("Fraction of Weighted Positive Cycles", fontweight = 'bold' , c='0.3')
plt.ylabel("Avg. Dynamic JSD\n(Param Variation)", fontweight = 'bold' , c='0.3')
#plt.title("Random Networks (All Sizes):        ρ = {:.3f}".format(corr), fontweight = 'bold' , c='0.3')
ax.legend(handles = test, prop={'size': 25})

f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)
plt.tight_layout()
plt.savefig("dynamicjsd_paramvar.jpg")
plt.clf()    
    


