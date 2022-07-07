from os.path import splitext, isfile, join
from os import listdir
import networkx as nx
import numpy as np
import modules.metric as metric
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import numpy as np
import scipy as sp
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
colorarr = []
col = ['r' , 'g', 'm', 'k' ,'c', 'y', 'C1' ]
import warnings
import copy
 
matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'30', 'axes.labelsize':'30', 'xtick.labelsize':'25', 'ytick.labelsize':'25', 'legend.fontsize':'25'})
plt.rcParams['figure.dpi'] = 500

y_label_arr = ["Avg. Perturbation JSD", "Avg. Fold Change in Plasticity\n(Structural)", "RACIPE vs Cont. (JSD)", "Avg. Fold Change in Plasticity\n(Dynamic)"]

y_leg_arr = ["pJSD", "pPLast", "dJSD", "dPlast"]

line_arr = ['solid' , 'dashed', 'dashdot', 'dotted']

size = 4
r=2
data  = [0,0,0,0]
for i in range(len(y_label_arr)):
    arr = np.loadtxt("{}_Size{}.txt".format(y_label_arr[i],size) )
    arr = arr.T
    arr[0] = np.abs(arr[0])
    arr[1] = np.abs(arr[1])
    data[i] = arr

fig,ax = plt.subplots()
for i in range(4):
    plt.plot(data[i][0], data[i][1] , linewidth = 4 , linestyle = line_arr[i])
    
plt.xlabel("Weight for WPFLs")
plt.ylabel("Absolute Correlation")
plt.title("Size 4" , x=0.5, y=0.93)
min1 = ax.get_ylim()
ymax = max(data[1][1]) + 0.1
plt.ylim([min1[0],ymax])

plt.legend(y_leg_arr,markerscale=2)
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)
plt.tight_layout()
plt.savefig("size4combined.png", transparent = True)
plt.clf()
matplotlib.rcParams.update({'legend.fontsize':'17'})
sizearr = [8,3,6,6]

for i in range(len(y_label_arr)):
    data  = [0]*sizearr[i]
    l = sizearr[i]-1
    legarr = []
    
    for j in range(4,4+l+1):
         jstar = j
         
         if(j==4+l):
            jstar = -1
            legarr.append("All sizes")
         else:
            legarr.append("Size {}".format(j))
         arr = np.loadtxt("{}_Size{}.txt".format(y_label_arr[i],jstar) )
         arr = arr.T
         arr[0] = np.abs(arr[0])
         arr[1] = np.abs(arr[1])
         data[j-4] = arr
    fig,ax = plt.subplots()
    
    for j in range(0,l+1):
        plt.plot(data[j][0], data[j][1] , linewidth = 4)
    
    plt.xlabel("Weight for WPFLs")
    plt.ylabel("Absolute Correlation")
    plt.title("{}".format(y_leg_arr[i]) , x=0.5, y=0.93)
    ymax = 0
    for j in range(0,l+1):
    
        ymax = max(max(data[j][1]) , ymax)
    ymax += 0.1
    min1 = ax.get_ylim()
    plt.ylim([min1[0],ymax])
    plt.legend(legarr,markerscale=2)
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)
    plt.tight_layout()
    plt.savefig("{}_sizescombined.png".format(y_leg_arr[i]), transparent = True)
    plt.clf()    