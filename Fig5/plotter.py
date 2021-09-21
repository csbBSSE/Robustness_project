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
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns

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

def topofiles_size(topofiles, size):
    if size == -1:
        topo_size = []
        for i in topofiles:
            lm = metric.matrix(i)
            #if lm.shape[0] != 5:
            topo_size.append(i)
        return topo_size
    topo_size = []
    for i in topofiles:
        lm = metric.matrix(i)
        if lm.shape[0] == size:
            topo_size.append(i)
    return topo_size

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size':'25', 'xtick.labelsize':'20', 'ytick.labelsize':'20', 'axes.labelsize':'25'})

topofiles= [splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()

kplast_dict = {}
kplast_file = open("raw_data/kinetic_plast.txt", 'r').readlines()
for i in kplast_file:
    j = i.split(" ")
    kplast_dict[j[0]] = float(j[1])

db_emp = {}
db_met = {}

for i in range(len(topofiles)):
    network_name = topofiles[i]

    g = metric.networkx_graph(network_name)
    cycle_info = metric.cycle_info(network_name, g)

    lm = metric.matrix(network_name)

    act_eig = metric.act_eig(network_name, g, lm)

    #structural
        #perturbation JSD

    try:
        if("fix" not in network_name):
            flag= 1/ 0
        pjsd = np.mean(np.loadtxt("raw_data/cnt2_{}_jsd.txt".format(network_name)))
    except:
        pjsd = -100

        #fold change

    try:
        data = np.loadtxt("raw_data/{}_plastdata.txt".format(network_name))
        fold_arr = []
        for a in data[1:]:
            foldchange = 0
            wt = data[0]

            if (wt == 0 and a != 0) or (wt != 0 and a == 0):
                foldchange = 0
            elif (wt == 0 and a == 0):
                foldchange = 1
            else:
                foldchange = min(wt / a, a / wt)
            fold_arr.append(foldchange)
        fchg = np.mean(fold_arr)
    except:
        fchg = -100

    #kinetic robustness
        #JSD b/w RACIPE and boolean
    try:
        ising = open("raw_data/{}_ising_probfull.txt".format(network_name), 'r').readlines()
        flag = 1
        racipe = np.loadtxt("raw_data/{}_racipe_probfull_processed.txt".format(network_name)).T

        isingdict = {}
        for i in range(2 ** lm.shape[0]):
            isingdict[i] = 0
        for j in ising:
            if j == "":
                continue
            i = j.split(" ")
            isingdict[int(i[0], 2)] = float(i[1])
        isingarr = np.array([isingdict[i] for i in range(2 ** lm.shape[0])])
        # print(isingarr)
        # print(racipe)
        kjsd = jensenshannon(racipe, isingarr, 2)
    except:
        kjsd = -100

        #Plasticity
    try:
        kplast = kplast_dict[network_name]
    except:
        kplast = -100

    db_emp[network_name] = [pjsd, fchg, kjsd, kplast]
    db_met[network_name] = [*cycle_info]

print("dictionaries built")

######################## PLOTTING ########################



corrarr = []
corrarrpjsd  =[]



def plotter(x_arr, y_arr, x_label, y_label, dir, name, size):
    global corrarr
    global corrarrpjsd
    
    if(x_label == "Fraction of Weighted Feedback Loops"):
        temp1 = []
        temp2 = []
        for i in range(len(y_arr)):
            temp1.append(x_arr[i][0])
            temp2.append(x_arr[i][1])
        p0 = 1.0 , 1.0 , 1.0
        y_arr = np.array(y_arr)
        x_arr1 = np.array(temp1)
        x_arr2 = np.array(temp2)
        try:
            popt, pcov = curve_fit(func, (x_arr1,x_arr2), y_arr, p0)
            z_arr = func_output( (x_arr1,x_arr2) , popt[0], popt[1] , popt[2])
            x_arr = np.array(z_arr)   
        except:
            return None
            
    if len(x_arr) == 0 or len(y_arr) == 0:
        return None
    if np.isnan(x_arr).any() or np.isnan(y_arr).any():
        return None
    if np.isinf(x_arr).any() or np.isinf(y_arr).any():
        return None
    
    r = 2

    fig, ax = plt.subplots()
    matplotlib.rcParams.update({'font.size': 10*r})


    sns.regplot(x_arr,y_arr,line_kws = {"color": 'r'})
    ax.lines.pop(0)
    plt.ylabel(y_label, fontweight="bold" , c='0.3' )
    plt.xlabel(x_label, fontweight="bold" , c='0.3' )
    lim1 = min(x_arr) - 0.05*(max(x_arr) - min(x_arr))
    lim2 = max(x_arr) + 0.05*(max(x_arr) - min(x_arr))
    ax.set(xlim=(lim1 , lim2))

    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)

    title = ""
    if (size == -1):
        title = "Random Networks (All Sizes)"
    else:
        title = "Random Networks (Size {})".format(size)

    try:
        pcorr, _ = pearsonr(x_arr,y_arr)
    except:
        print(x_arr, y_arr, x_label, y_label, dir, name, size)
          
    if (size == -1):
        corrarr.append(np.round(pcorr*10000)/10000)
        
    if (y_label == "Average Perturbation JSD from WT"):
        print(size)
        corrarrpjsd.append(np.round(pcorr*10000)/10000)        
        
    # print(pcorr)
    plt.title(title + "    ρ = {:.3f}".format(pcorr), fontweight="bold", c = '0.3')
    plt.tight_layout()
    plt.savefig("plots/{}/{}.jpg".format(dir, name))
    # print("{}/{}.jpg".format(dir, name))
    plt.close()

def plotterscript(topofiles_all, db_emp, db_met, x_label_arr, y_label_arr, dir_arr, met_arr):
    size_arr = [4, 5, 6, 7, 8, 9, 10 , -1]
    for size in size_arr:
        topofiles = topofiles_size(topofiles_all, size)
        print("Size:", size)
        for emp_index in range(len(y_label_arr)):
            for met_index in range(len(x_label_arr)):
                    x_arr = []
                    y_arr = []
                    for i in topofiles:
                        if db_emp[i][emp_index] != -100 and db_met[i][met_index] != -100:
                            if(x_label_arr[met_index]!="Fraction of Weighted Feedback Loops" ):
                                #print(db_emp[i][emp_index] , db_met[i][met_index] , x_label_arr[met_index])
                                if np.isfinite(db_emp[i][emp_index]) and np.isfinite(db_met[i][met_index]):
                                    x_arr.append(db_met[i][met_index])
                                    y_arr.append(db_emp[i][emp_index])
                            else:
                                if np.isfinite(db_emp[i][emp_index]) and (np.isfinite(db_met[i][met_index][0] and np.isfinite(db_met[i][met_index][1]))):
                                    x_arr.append(db_met[i][met_index])
                                    y_arr.append(db_emp[i][emp_index])                                
                    size_str = "all" if size == -1 else str(size)
                    name = "{}_{}_{}".format(dir_arr[emp_index], met_arr[met_index], size_str)
                    # print(x_arr, y_arr)
                    plotter(x_arr, y_arr, x_label_arr[met_index], y_label_arr[emp_index], dir_arr[emp_index], name, size)                   

x_label_arr = ["Number of Positive Feedback Loops", "Number of Negative Feedback Loops", "Fraction of Positive Cycles", "Fraction of Weighted Feedback Loops"]
y_label_arr = ["Average Perturbation JSD from WT", "Average Fold Change (Plasticity) from WT", "JSD between RACIPE and Cont.", "Kinetic Robustness in Plasticity"]
dir_arr = ["pjsd", "fchg", "kjsd", "kplast"]
met_arr = ["npos", "nneg", "fracpos" ,"wfracloops"]
plotterscript(topofiles, db_emp, db_met, x_label_arr, y_label_arr, dir_arr, met_arr)

print("corrarr")
print(corrarr)

print("corrarrpjsd")
print(corrarrpjsd)


x_labels = ["PFL", "NFL" , "FPL" , "WFPL" ]
y_labels = ["PJSD" , "fchg" , "kjsd" , "kplast"]

data = corrarr

data = np.abs(np.array(data))
data = data.reshape((4,4))

r = 2
fig, ax = plt.subplots()   
matplotlib.rcParams.update({'font.size': 10*r})  
sns.heatmap(data , xticklabels = x_labels , yticklabels = y_labels, annot = True)  
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)
plt.tight_layout()

plt.savefig("heatmap.png")
plt.clf()

import numpy as np
import seaborn as sns


x_labels = ["NPL", "NNL", "FPL" , "WFPL"]
y_labels = ["4" , "5","6", "7" , "8" , "9" , "10" , "All"]


r = 2
fig, ax = plt.subplots()   
matplotlib.rcParams.update({'font.size': 10*r})  
sns.mpl_palette("jet", 6)
data = corrarrpjsd
data = np.abs(np.array(data))
data = data.reshape((8,4))
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)

data1 = data.T
x1 = [4 , 5, 6 ,7 ,8 , 9 , 10 , "All"]
import matplotlib.pyplot as plt

plt.plot(x1, data1[0] ,  marker='o')
plt.plot(x1, data1[1] ,  marker="v")
plt.plot(x1, data1[2] , marker = "p")
plt.plot(x1, data1[3] , marker = "D")
plt.xlabel("Random Network Size" , c='0.3', fontweight = 'bold')
plt.ylabel("Correlation with Avg. Perturbation JSD")
plt.legend(['Number of Positive Cycles' , 'Number of Negative Cycles', 'Fraction of Positive Cycles' ,'Weighted Fraction of Positive Cycles'])
plt.tight_layout()
plt.ylim([0,1.1])

plt.savefig("lineplot.png")
plt.clf()
