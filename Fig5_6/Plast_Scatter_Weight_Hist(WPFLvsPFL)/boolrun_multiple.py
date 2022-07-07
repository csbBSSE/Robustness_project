import os
import time
from os import listdir
from os.path import isfile, join
from os.path import splitext, isfile, join
from os import listdir
import networkx as nx
import numpy as np
import initialise.initialise as initialise
import initialise.parser as parser
import modules.metric as metric
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import seaborn
from scipy.optimize import curve_fit
print('imported modules')
plt.rcParams['figure.dpi'] = 500

def starfunc(significance):
    if significance < 0.001:
        return "***"
    elif significance < 0.01:
        return "**"
    elif significance < 0.05:
        return "*"
    else:
        return ""

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()

def func_delta(X, a , b , c):
    x,y = X
    z1 = [c]*len(x)
    z1 = np.array(z1)
    z = a*x - y
    return np.array(b*z+ z1)

def func_delta_output(X, a , b , c):
    x,y = X
    z = a*x - y
    return np.array(z)

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'40', 'axes.labelsize':'35', 'xtick.labelsize':'33', 'ytick.labelsize':'30', 'legend.fontsize':'30'})

print(topofiles)
####
import initialise.initialise as initialise
import initialise.parser as parser
in_file = 'init.txt'
begin=1
process_count=1
params = initialise.initialise(in_file)

id_to_node=[]
link_matrix = [0]*len(topofiles)
copy_linkmatrix = [0]*len(topofiles)
id_to_node = [0]*len(topofiles)
length = len(topofiles)
nodes = [0]*len(topofiles)
for j in range (len(topofiles)):
                repj=topofiles[j]
                random_seed = int(begin) + process_count
                print(topofiles[j])
                link_matrix[j], id_to_node[j] = parser.parse_topo(params,repj, random_seed)
                copy_linkmatrix[j], id_to_node[j] = parser.parse_topo(params,repj, random_seed)
                nodes[j] = len(id_to_node[j])

num_simulations = 10000
num_threads = 5

inittext = """input_folder_name input
output_folder_name output
input_filenames {}
num_runs 1
num_simulations {}
maxtime 2000
constant_node_count 0
"""


                 

corrarr1=[]
corrarr2=[]
corrarr3=[]
corrarr4=[]
corrarr5=[]
for i in range(len(topofiles)):

    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []
    
    network_name = network_name = topofiles[i]
    jsd_data = np.loadtxt("raw_data/{}_plastdata.txt".format(network_name) )[1:]
    
    print(i)
    curnetwork = open("curnetwork.txt", 'w')
    looptime = time.time()
    initfile = open("init.txt", "w")
    initfile.write(inittext.format(topofiles[i], num_simulations * nodes[i]))
    initfile.close()
    outdegarr=[0]*nodes[i]
    indegarr=[0]*nodes[i]

    for j in range(nodes[i]):
        for k in range(nodes[i]):
            if(copy_linkmatrix[i][j][k] !=0):
                outdegarr[j]+=1
                
            if(copy_linkmatrix[i][k][j] !=0):
                indegarr[j]+=1
    
    counter = 0
    for j in range(nodes[i]):
        for k in range(nodes[i]):
            for l in range(3):
                if l-1 == copy_linkmatrix[i][j][k]:
                    continue
                else:
                    if(copy_linkmatrix[i][j][k]!=0 and ( indegarr[j]+outdegarr[j] ==1 or  indegarr[k]+outdegarr[k] ==1)):
                        if(l-1 == 0):
                            continue
                    
                    network_name = topofiles[i]
                    lm = link_matrix[i]
                    
                    g = metric.networkx_graph(network_name)
                    cycle_info = metric.cycle_info(network_name, g)
                   
                    a_1 = cycle_info[0] 
                    a_2 = cycle_info[1]
                    a_3 = cycle_info[5]
                    
                    link_matrix[i][j][k] = l - 1
                    tempfile = "{}_{}_{}_{}".format(topofiles[i], j, k, l)
                    initfile = open("init.txt", "w")
                    initfile.write(inittext.format(tempfile, num_threads,nodes[i]*num_simulations))
                    initfile.close()
                    temptopofile = open("input/" + tempfile + ".topo", 'w')
                    
                    tempidsfile = open("input/" + tempfile + ".ids", 'w')
                    tempidsfile.write(open("input/" +topofiles[i]+".ids", 'r').read())
                    tempidsfile.close()

                    strtemp = "Source Target Type\n"
                    for p1 in range(nodes[i]):
                        for p2 in range(nodes[i]):
                            #print(p1,p2)
                            if link_matrix[i][p1][p2] == 0:
                                continue
                            else:
                                strtemp += id_to_node[i][p1] + " " + id_to_node[i][p2] + " " + str(1 if link_matrix[i][p1][p2] == 1 else 2) + "\n"
                    
                    strtemp = strtemp[:-1]
                    temptopofile.write(strtemp)
                    temptopofile.close()
                    
                    network_name = tempfile
                    
                    lm = link_matrix[i]
                    g = metric.networkx_graph(network_name)
                    cycle_info = metric.cycle_info(network_name, g)
                  
                    a_4 = cycle_info[0] 
                    a_5 = cycle_info[1]
                    a_6 = cycle_info[5]
                    
                    
                    arr1.append(a_4-a_1)
                    arr2.append(a_5-a_2)
                    arr3.append(a_6-a_3)
         
                    counter+=1
                    link_matrix[i][j][k] = copy_linkmatrix[i][j][k]
                    os.remove("input/" + tempfile + ".ids")
                    os.remove("input/" + tempfile + ".topo")

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    arr3 = np.array(arr3)
    save = 0
    network_name = topofiles[i]
    r = 2
    fig,ax = plt.subplots()  
    pcorr, significance = pearsonr(arr1 , jsd_data)    
    seaborn.regplot(arr1, jsd_data , line_kws = {"color": 'b'} )  
    ax.lines.pop(0)    
    ax.spines['top'].set_visible(False)
    plt.xlabel("Δ PFLs",fontweight="bold" , c='0.3')
    plt.ylabel("Plasticity",fontweight="bold" , c='0.3')   
    plt.title("{}\nPerturbations".format(network_name),fontweight="bold" , c='0.3', x = 0.65, y  = 0.9)
    corrarr1.append(pcorr)  
    textstr = r'$\mathrm{ρ}=%.3f$' % (pcorr, )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    save = pcorr
    ax.text(0.10, 0.95, textstr + starfunc(significance), transform=ax.transAxes, fontsize=35,
        verticalalignment='top', bbox=props)
        
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)    
    
    plt.tight_layout()
    plt.savefig("img/{}_plast_pos.png".format(topofiles[i]), transparent = True)   
    plt.clf()
    plt.close(fig)
    ###################   
    fig,ax = plt.subplots()  
    pcorr, significance = pearsonr(arr2 , jsd_data)    
    seaborn.regplot(arr2, jsd_data , line_kws = {"color": 'b'} )  
    ax.lines.pop(0)    
    ax.spines['top'].set_visible(False)
    plt.xlabel("Δ NFLs",fontweight="bold" , c='0.3')
    plt.ylabel("Plasticity",fontweight="bold" , c='0.3')   
    plt.title("{}\nPerturbations".format(network_name),fontweight="bold" , c='0.3', x = 0.65, y  = 0.9)

    textstr = r'$\mathrm{ρ}=%.3f$' % (pcorr, )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.10, 0.95, textstr + starfunc(significance), transform=ax.transAxes, fontsize=35,
        verticalalignment='top', bbox=props)
        
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)    
    
    plt.tight_layout()
    plt.savefig("img/{}_plast_neg.png".format(topofiles[i]), transparent = True)   
    plt.clf()
    plt.close(fig)
    ##################
    fig,ax = plt.subplots()  
    pcorr, significance = pearsonr(arr3 , jsd_data)    
    seaborn.regplot(arr3, jsd_data , line_kws = {"color": 'b'} )  
    ax.lines.pop(0)    
    ax.spines['top'].set_visible(False)
    plt.xlabel("Δ Weighted PFLs (length)",fontweight="bold" , c='0.3')
    plt.ylabel("Plasticity",fontweight="bold" , c='0.3')   
    plt.title("{}\nPerturbations".format(network_name),fontweight="bold" , c='0.3', x = 0.65, y  = 0.9)
    corrarr2.append(pcorr)  
    textstr = r'$\mathrm{ρ}=%.3f$' % (pcorr, )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.10, 0.95, textstr + starfunc(significance), transform=ax.transAxes, fontsize=35,
        verticalalignment='top', bbox=props)
        
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)    
    
    plt.tight_layout()
    plt.savefig("img/{}_plast_pos_length.png".format(topofiles[i]), transparent = True)   
    plt.clf()
    plt.close(fig)
    
    ######
 ##################
    fig,ax = plt.subplots()  
    pcorr, significance = pearsonr(arr1 + arr2 , jsd_data)    
    seaborn.regplot(arr1 + arr2, jsd_data , line_kws = {"color": 'b'} )  
    ax.lines.pop(0)    
    ax.spines['top'].set_visible(False)
    plt.xlabel("Δ FLs",fontweight="bold" , c='0.3')
    plt.ylabel("Plasticity",fontweight="bold" , c='0.3')   
    plt.title("{}\nPerturbations".format(network_name),fontweight="bold" , c='0.3', x = 0.65, y  = 0.9)
    textstr = r'$\mathrm{ρ}=%.3f$' % (pcorr, )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.10, 0.95, textstr + starfunc(significance), transform=ax.transAxes, fontsize=35,
        verticalalignment='top', bbox=props)
        
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)    
    
    plt.tight_layout()
    plt.savefig("img/{}_plast_tot.png".format(topofiles[i]), transparent = True)   
    plt.clf()
    plt.close(fig)
    
    ######    
    
    
    flag = 0
    r = 2
    fig = plt.figure()
    
    p0 = 1.0 , 1.0 , 1.0
    y_arr = np.array(jsd_data)
    x_arr1 = np.array(arr1)
    x_arr2 = np.array(arr2)  
    z_arr = arr1 
    try:
        popt, pcov = curve_fit(func_delta, (x_arr1,x_arr2), y_arr, p0)
        z_arr = func_delta_output( (x_arr1,x_arr2) , popt[0], popt[1] , popt[2])
        z_arr = np.array(z_arr)      
    except:
        flag = 1
        pass
        
        
    fig,ax = plt.subplots()  
    pcorr, significance = pearsonr(z_arr , jsd_data)    
    seaborn.regplot(z_arr, jsd_data , line_kws = {"color": 'b'} )  
    ax.lines.pop(0)    
    ax.spines['top'].set_visible(False)
    plt.xlabel("Δ Weighted Cycle Sum (sign)",fontweight="bold" , c='0.3')
    plt.ylabel("Plasticity",fontweight="bold" , c='0.3')   
    plt.title("{}\nPerturbations".format(network_name),fontweight="bold" , c='0.3', x = 0.65, y  = 0.9)
    textstr = r'$\mathrm{ρ}=%.3f$' % (pcorr, )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.10, 0.95, textstr + starfunc(significance), transform=ax.transAxes, fontsize=35,
        verticalalignment='top', bbox=props)
        
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)    
    
    plt.tight_layout()
    plt.savefig("img/{}_plast_cycle_sum_weighted.png".format(topofiles[i]), transparent = True)   
    plt.clf()
    plt.close(fig)
    pcorr1 = pcorr
 
    
    ###########
    fig,ax = plt.subplots()  
    pcorr, significance = pearsonr(arr1 - arr2 , jsd_data)    
    seaborn.regplot(arr1 - arr2, jsd_data , line_kws = {"color": 'b'} )  
    ax.lines.pop(0)    
    ax.spines['top'].set_visible(False)
    plt.xlabel("Δ Cycle Sum",fontweight="bold" , c='0.3')
    plt.ylabel("Plasticity",fontweight="bold" , c='0.3')   
    plt.title("{}\nPerturbations".format(network_name),fontweight="bold" , c='0.3', x = 0.65, y  = 0.9)
    textstr = r'$\mathrm{ρ}=%.3f$' % (pcorr, )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.10, 0.95, textstr + starfunc(significance), transform=ax.transAxes, fontsize=35,
        verticalalignment='top', bbox=props)
        
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)    
    
    plt.tight_layout()
    plt.savefig("img/{}_plast_cycle_sum.png".format(topofiles[i]), transparent = True)   
    plt.clf()
    plt.close(fig)
    if(pcorr1 > 0 and flag == 0):
    
        corrarr3.append(pcorr)
    
        corrarr4.append(pcorr1)

        corrarr5.append(save)




fig, ax = plt.subplots()   
r = 2
    

ax.scatter(corrarr3,corrarr4)

plt.xlabel("ρ (Plasticity vs Δ Cycle Sum)",fontweight="bold" , c='0.3')
plt.ylabel("ρ (Plasticity vs Δ WFLs)(sign)",fontweight="bold" , c='0.3')   
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax.plot(lims, lims, 'k-', zorder=0 , linewidth = 3)
plt.tight_layout()
plt.savefig("img/weightedorno_cyclesum.png", transparent = True)     
plt.clf()
plt.close(fig)
fig, ax = plt.subplots()   
r = 2
###################

fig, ax = plt.subplots()   
r = 2
    

ax.scatter(corrarr5,corrarr4)

plt.xlabel("ρ (Plasticity vs Δ PFLs)",fontweight="bold" , c='0.3')
plt.ylabel("ρ (Plasticity vs Δ WPFLs)",fontweight="bold" , c='0.3')   
##

f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax.plot(lims, lims, 'k-', zorder=0 , linewidth = 3)
plt.tight_layout()
plt.savefig("imgweightedorno_length.png", transparent = True)     
plt.clf()
plt.close(fig)
fig, ax = plt.subplots()   
r = 2
        
############
ax.scatter(corrarr1,corrarr2)

plt.xlabel("ρ (Plasticity vs Δ PFLs)",fontweight="bold" , c='0.3')
plt.ylabel("ρ (Plasticity vs Δ SWFLs)",fontweight="bold" , c='0.3') 
##

f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)



lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax.plot(lims, lims, 'k-', zorder=0 , linewidth = 3)
plt.tight_layout()
plt.savefig("img/weightedorno_cyclesum_pos.png", transparent = True)     
plt.close(fig)
