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
 
warnings.filterwarnings("ignore")
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
    if(a<0):
        return -10000000000000
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

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'30', 'axes.labelsize':'30', 'xtick.labelsize':'25', 'ytick.labelsize':'25', 'legend.fontsize':'25'})

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


    #structural
        #perturbation JSD

    try:
        if("fix" not in network_name and network_name[0]=='r'):
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
        #JSD b/w RACIPE and cont
    try:
        cont = open("raw_data/{}_ising_probfull.txt".format(network_name), 'r').readlines()    ## note: ising_probfull is actually contdata, not ising
        flag = 1
        racipe = np.loadtxt("raw_data/{}_racipe_probfull_processed.txt".format(network_name)).T

        contdict = {}
        for i in range(2 ** lm.shape[0]):
            contdict[i] = 0
        for j in cont:
            if j == "":
                continue
            i = j.split(" ")
            contdict[int(i[0], 2)] = float(i[1])
        contarr = np.array([contdict[i] for i in range(2 ** lm.shape[0])])
        # print(contarr)
        # print(racipe)
        kjsd = jensenshannon(racipe, contarr, 2)
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
corrarrpplast =[]
corrarrdjsd  =[]
corrarrdplast =[]
weightarrpjsd  =[]
weightarrpplast =[]
weightarrdjsd  =[]
weightarrdplast =[]

def plotter(topofiles, x_arr, y_arr, x_label, y_label, dir, name, size):
    global corrarr
    global corrarrpjsd
    global corrarrpplast
    global corrarrdjsd
    global corrarrdplast
    global weightarrpjsd
    global weightarrdjsd
    global weightarrpplast
    global weightarrdplast
    
    weight = 0
    
    if(x_label == "Fraction of Weighted Positive Cycles"):
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
            weight = popt[0]
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

    
    sns.regplot(x_arr,y_arr,line_kws = {"color": 'b'} )
    
    test = []
    

            
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
        pcorr, significance = pearsonr(x_arr,y_arr)
    except:
        print(x_arr, y_arr, x_label, y_label, dir, name, size)
    
    #y_label_arr = ["Avg. Perturbation JSD", "Avg. Fold Change in Plasticity\n(Structural)", "RACIPE vs Cont. (JSD)", "Avg. Fold Change in Plasticity\n(Dynamic)"]    
    if (size == -1 and x_label!=  "No. of FLs" ):
        corrarr.append(np.round(pcorr*10000)/10000)
        
    if (y_label == "Avg. Fold Change in Plasticity\n(Structural)" and x_label!=  "No. of FLs"):
        corrarrpplast.append(np.round(pcorr*10000)/10000)        
        if(x_label == "Fraction of Weighted Positive Cycles"):
            weightarrpplast.append(weight)
 
    if (y_label == "Avg. Perturbation JSD" and x_label!=  "No. of FLs" ):
 
        corrarrpjsd.append(np.round(pcorr*10000)/10000)  
        if(x_label == "Fraction of Weighted Positive Cycles"):
            weightarrpjsd.append(weight)
            
    if (y_label == "RACIPE vs Cont. (JSD)" and x_label!=  "No. of FLs"):

        corrarrdjsd.append(np.round(pcorr*10000)/10000)  
        if(x_label == "Fraction of Weighted Positive Cycles"):
            weightarrdjsd.append(weight)       
    if (y_label == "Avg. Fold Change in Plasticity\n(Dynamic)" and x_label!=  "No. of FLs"):

        corrarrdplast.append(np.round(pcorr*10000)/10000)      
        if(x_label == "Fraction of Weighted Positive Cycles"):
            weightarrdplast.append(weight)
        

     
        
    if(size!= 15):
        colarr = []
        count = 0
        x_arr1 = []
        y_arr1 = []
        labels = []
        test.append(Line2D([], [], color= 'w', marker='X',
                          markersize=20, label= "ρ = {:.3f}{}".format(pcorr, starfunc(significance)) ,linestyle='None'))
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
            
    # print(pcorr)
    #plt.title(title + "    ρ = {:.3f}".format(pcorr), fontweight="bold", c = '0.3
    #leg1 = ax.legend(handles = test, prop={'size': 25})
    
    #leg2 = ax.legend(handles = [Line2D([0], [0], marker='o', color='w', label="ρ = {:.3f}{}".format(pcorr, starfunc(significance)), markerfacecolor='w', markersize=5)])
    #ax.add_artist(leg1)

    ax.legend(handles = test, prop={'size': 25})
    plt.tight_layout()
    plt.savefig("plots/{}/{}.png".format(dir, name), transparent = True)
    # print("{}/{}.jpg".format(dir, name))
    plt.close()

def plotterscript(topofiles_all, db_emp, db_met, x_label_arr, y_label_arr, dir_arr, met_arr):
    size_arr = [4, 5, 6, 7, 8, 9, 10 , -1]
    for size in size_arr:
        topofiles = topofiles_size(topofiles_all, size)

        print("Size:", size)
        for emp_index in range(len(y_label_arr)):
            for met_index in range(len(x_label_arr)):
                    topofile_input = []
                    x_arr = []
                    y_arr = []
                    for i in topofiles:
                        if db_emp[i][emp_index] != -100 and db_met[i][met_index] != -100:
                            if(x_label_arr[met_index]!="Fraction of Weighted Positive Cycles" ):
                                #print(db_emp[i][emp_index] , db_met[i][met_index] , x_label_arr[met_index])
                                if np.isfinite(db_emp[i][emp_index]) and np.isfinite(db_met[i][met_index]):
                                    x_arr.append(db_met[i][met_index])
                                    y_arr.append(db_emp[i][emp_index])
                                    topofile_input.append(i)
                            else:
                                if np.isfinite(db_emp[i][emp_index]) and (np.isfinite(db_met[i][met_index][0] and np.isfinite(db_met[i][met_index][1]))):
                                    x_arr.append(db_met[i][met_index])
                                    y_arr.append(db_emp[i][emp_index])      
                                    topofile_input.append(i)                                        
                    size_str = "all" if size == -1 else str(size)
                    name = "{}_{}_{}".format(dir_arr[emp_index], met_arr[met_index], size_str)
                    topofile_input = list(map(lambda x: x if x != 'OVOL2' else 'OVOL', topofile_input))
                    topofile_input.sort()
                    #print("input", topofile_input[0])
                    # print(x_arr, y_arr)
                    plotter(topofile_input, x_arr, y_arr, x_label_arr[met_index], y_label_arr[emp_index], dir_arr[emp_index], name, size)                   

x_label_arr = ["No. of PFLs", "No. of NFLs", "No. of FLs" ,"Fraction of Positive Cycles", "Fraction of Weighted Positive Cycles"]
y_label_arr = ["Avg. Perturbation JSD", "Avg. Fold Change in Plasticity\n(Structural)", "RACIPE vs Cont. (JSD)", "Avg. Fold Change in Plasticity\n(Dynamic)"]
dir_arr = ["pjsd", "fchg", "djsd", "dplast"]
met_arr = ["npos", "nneg", "totfl", "fracpos" ,"wfracloops"]
plotterscript(topofiles, db_emp, db_met, x_label_arr, y_label_arr, dir_arr, met_arr)

print("corrarr")
print(corrarr)

print("corrarrpjsd")
print(corrarrpjsd)


x_labels = ["PFL", "NFL", "FPC" , "FWPC"]
y_labels = ["pJSD" , "pPlast" , "dJSD" , "dplast"]

data = corrarr

data = np.abs(np.array(data))
data = data.reshape((4,4))

r = 2
fig, ax = plt.subplots()   
heatmap = sns.heatmap(data , xticklabels = x_labels , yticklabels = y_labels, annot = True)  
heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize = 30)
heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize = 30)
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)
plt.tight_layout()

plt.savefig("heatmap.png", transparent = True)
plt.clf()

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def lineplotter(data,y_label):
    x_labels = ["PFL", "NFL", "FPC" , "FWPC"]
    l1 = len(data)//4 -1
    if(y_label == "Avg. Fold Change in Plasticity\n(Structural)"):
        l1 = 2
    for j in range(4,4+l1):
        x_labels.append("{}".format(j))
    x_labels.append("All")
    
    r = 2
    fig, ax = plt.subplots()   
    sns.mpl_palette("jet", 6)
    data = np.abs(np.array(data))
    data = data.reshape((l1+1,4))
    
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf() 
    fig.set_size_inches(f)

    data1 = data.T
    
    x1 = []
    for j in range(4,4+l1):
        x1.append(j)
    
    plt.plot(x1, data1[0][:-1] ,  marker='o' , linewidth = 4)
    plt.plot(x1, data1[1][:-1] ,  marker="o", linewidth = 4)
    plt.plot(x1, data1[2][:-1] , marker = "o", linewidth = 4)
    plt.plot(x1, data1[3][:-1] , marker = "o", linewidth = 4)
    arrcol =  ['C0','C1','C2','C3']
    for j in range(4):
        plt.plot( 4+l1, data1[j][-1] , marker = 'X', markersize = 20, c = arrcol[j] ,linestyle = 'None' )
    x2 = copy.deepcopy(x1)
    #x2.insert(0,"0")
    x2.append("All")
    for j in range(len(x2)):
        x2[j] = str(x2[j])
    print(x2)
    ax.set_xticklabels(x2)
    plt.xlabel("Random Network Size" , c='0.3', fontweight = 'bold')
    plt.ylabel(y_label)
    plt.legend(['Number of Positive Cycles' , 'Number of Negative Cycles', 'Fraction of Positive Cycles(Unweighted)' ,'Fraction of Weighted Positive Cycles'], fancybox = True)
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)
    plt.tight_layout()
    plt.ylim([-0.03,1.1])
    if(y_label == "Avg. Fold Change in Plasticity\n(Dynamic)"):
         plt.ylim([-0.03,1.25])

    plt.savefig("lineplot_{}.png".format(y_label), transparent = True)
    plt.clf()
    
#y_label_arr = ["Avg. Perturbation JSD", "Avg. Fold Change in Plasticity\n(Structural)", "RACIPE vs Cont. (JSD)", "Avg. Fold Change in Plasticity\n(Dynamic)"]   
    
lineplotter(corrarrpjsd,"Avg. Perturbation JSD")
lineplotter(corrarrpplast,"Avg. Fold Change in Plasticity\n(Structural)")
lineplotter(corrarrdjsd,"RACIPE vs Cont. (JSD)")
lineplotter(corrarrdplast,"Avg. Fold Change in Plasticity\n(Dynamic)")

np.savetxt("weightpjsd.txt" , weightarrpjsd , fmt='%1.3f')
np.savetxt("weightpplast.txt" , weightarrpplast , fmt='%1.3f')
np.savetxt("weightdjsd.txt" , weightarrdjsd, fmt='%1.3f')
np.savetxt("weightdplast.txt" , weightarrdplast, fmt='%1.3f')

'''  



    
x_labels = ["PFL", "NFL", "FPC" , "FWPC"]
y_labels = ["4" , "5","6", "7" , "8" , "9" , "10" , "All"]


r = 2
fig, ax = plt.subplots()   
sns.mpl_palette("jet", 6)
data = corrarrpjsd
data = np.abs(np.array(data))
data = data.reshape((8,4))
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)

data1 = data.T
x1 = [4 , 5, 6 ,7 ,8 , 9 , 10]
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.plot(x1, data1[0][:-1] ,  marker='o' , linewidth = 4)
plt.plot(x1, data1[1][:-1] ,  marker="o", linewidth = 4)
plt.plot(x1, data1[2][:-1] , marker = "o", linewidth = 4)
plt.plot(x1, data1[3][:-1] , marker = "o", linewidth = 4)
arrcol =  ['C0','C1','C2','C3']
for j in range(4):
     plt.plot( 11, data1[j][-1] , marker = 'X', markersize = 20, c = arrcol[j] ,linestyle = 'None' )

ax.set_xticklabels([0,4,5,6,7,8,9,10,"ALL"])
plt.xlabel("Random Network Size" , c='0.3', fontweight = 'bold')
plt.ylabel("Correlation with\nAvg. Perturbation JSD")
plt.legend(['Number of Positive Cycles' , 'Number of Negative Cycles', 'Fraction of Positive Cycles(Unweighted)' ,'Fraction of Weighted Positive Cycles'], fancybox = True)
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)
plt.tight_layout()
plt.ylim([-0.03,1.1])

plt.savefig("lineplot.png", transparent = True)
plt.clf()
'''