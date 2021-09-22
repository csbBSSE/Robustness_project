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
print('imported modules')



topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()



matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '25', 'axes.titlesize':'25', 'axes.labelsize':'25', 'xtick.labelsize':'20', 'ytick.labelsize':'20', 'legend.fontsize':'20'})

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


for i in range(len(topofiles)):
    wfndiffarr =[]
    wfndiffarr2 =[]
    network_name = network_name = topofiles[i]
    jsd_data = np.loadtxt("raw_data/{}_plastdata.txt".format(network_name) )[1:]
    
    print(i)
    curnetwork = open("curnetwork.txt", 'w')
    looptime = time.time()
    initfile = open("init.txt", "w")
    initfile.write(inittext.format(topofiles[i], num_simulations * nodes[i]))
    initfile.close()
    origwfn = 0
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
                   
                    fwn_1 = cycle_info[0] 
                    fwn_3 = cycle_info[5] 
                    
                    origwfn = fwn_1
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
                  
                    fwn_2 = cycle_info[0]
                    fwn_4 = cycle_info[5]
                    
                    wfndiffarr.append((fwn_2 - fwn_1))
                    wfndiffarr2.append((fwn_4 - fwn_3))
                    #print(fwn_1,fwn_2, fwn_2-fwn_1, jsd_data[counter])
                    counter+=1
                    link_matrix[i][j][k] = copy_linkmatrix[i][j][k]
                    os.remove("input/" + tempfile + ".ids")
                    os.remove("input/" + tempfile + ".topo")

    print(len(jsd_data))
    print(len(wfndiffarr))
    
    network_name = topofiles[i]
    r = 2
    fig,ax = plt.subplots()
    matplotlib.rcParams.update({'font.size': 10*r})    
    pcorr, _ = pearsonr(wfndiffarr , jsd_data)    
    seaborn.regplot(wfndiffarr, jsd_data , line_kws = {"color": 'r'} )  
    ax.lines.pop(0)    
    plt.xlabel("Δ Positive Cycle Sum",fontweight="bold" , c='0.3')
    plt.ylabel("Plasticity of Perturbed Network",fontweight="bold" , c='0.3')   
    plt.title("{}".format(network_name),fontweight="bold" , c='0.3' )
    
    textstr = r'$\mathrm{ρ}=%.2f$' % (pcorr, )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.10, 0.95, textstr, transform=ax.transAxes, fontsize=25,
        verticalalignment='top', bbox=props)
        
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)    

    plt.savefig("img/{}_plast_pos.jpg".format(topofiles[i]) )   
    plt.clf()
    
    
    r = 2
    fig = plt.figure()
    matplotlib.rcParams.update({'font.size': 10*r})    
    
    corrarr1.append(pcorr)
    pcorr, _ = pearsonr(wfndiffarr2 , jsd_data)    
    plt.scatter(wfndiffarr2, jsd_data )
    
 
    plt.xlabel("Δ Weighted Cycle Sum",fontweight="bold" , c='0.3')
    plt.ylabel("Plasticity of Perturbed Network",fontweight="bold" , c='0.3')   
    
    plt.title("{}  ρ = {:.3f}".format(network_name,pcorr),fontweight="bold" , c='0.3' )
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)        
    
    plt.savefig("img/{}_plast_weighted_neg.jpg".format(topofiles[i]) )   
    plt.clf()
    corrarr2.append(pcorr)

fig, ax = plt.subplots()   
r = 2
matplotlib.rcParams.update({'font.size': 10*r})    
    

ax.scatter(corrarr1,corrarr2)

plt.xlabel("Correlation vs Δ Cycle Sum",fontweight="bold" , c='0.3')
plt.ylabel("Correlation vs Δ Weighted Cycle Sum",fontweight="bold" , c='0.3')   
plt.title("Plasticty of Perturbed Networks:", fontweight="bold" , c='0.3' )
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.savefig("img/weightedorno_jsd_pos.jpg")     
