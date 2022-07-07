import os
import time
from os import listdir
from os.path import splitext, isfile, join
import networkx as nx
import math
import numpy as np
import initialise.initialise as initialise
import initialise.parser as parser
import modules.metric as metric
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from copy import copy
import matplotlib.colors as colors
from scipy.optimize import curve_fit
print('imported modules')

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'35', 'axes.labelsize':'38', 'xtick.labelsize':'28', 'ytick.labelsize':'19', 'legend.fontsize':'30'})

'''
def func_frac(X, a , b , c):    #custom function for fraction of cycles
    x,y = X
    z = np.zeros(len(x))
    z1 = [c]*len(x)
    z1 = np.array(z1)
    for i in range(len(x)):
        if(x[i] == 0 and y[i] == 0):
          z[i] = 0
        else:
            try:
                z[i] = a*x[i]/(a*x[i] + y[i])
            except:
                z[i] = 0
    return b*z + z1
'''  
 
def starfunc(significance):
    if significance < 0.001:
        return "***"
    elif significance < 0.01:
        return "**"
    elif significance < 0.05:
        return "*"
    else:
        return ""

def func_frac_output(X, a , b , c):    #custom function for fraction of cycles
    x,y = X
    z = np.zeros(len(x))
    for i in range(len(x)):
        if(x[i] == 0 and y[i] == 0):
          z[i] = 0
        else:
            try:
                z[i] = a*x[i]/(a*x[i] + y[i])
            except:
          
                z[i] = 0
    return z
    
    
'''
def fit_frac(x_arr,y_arr, target_arr):
  p0 = 3.0 , 1.0 , 1.0
  y_arr = np.array(y_arr)
  x_arr = np.array(x_arr)
  
  popt, pcov = curve_fit(func_frac, (x_arr,y_arr), target_arr, p0)
  popt = np.array(popt)

  z_arr = func_frac_output( (x_arr, y_arr) , popt[0], popt[1] , popt[2])
  return z_arr , popt
'''
'''
def func_diff(X, a , b , c):    #custom function for difference in cycles
    x,y = X
    z = np.zeros(len(x))
    z1 = [c]*len(x)
    z1 = np.array(z1)
    z = b*(a*x - y) + z1
    return z

def func_diff_output(X, a , b , c):    #custom function for difference in cycles
    x,y = X
    z = (a*x - y) 
    return np.array(z)
    
def fit_diff(x_arr,y_arr, target_arr):
  p0 = 3.0 , 1.0 , 1.0
  y_arr = np.array(y_arr)
  x_arr = np.array(x_arr)
  popt, pcov = curve_fit(func_diff, (x_arr,y_arr), target_arr, p0)
  popt = np.array(popt)
  

  z_arr = func_diff_output( (x_arr, y_arr) , popt[0], popt[1] , popt[2])

  return z_arr , popt
'''

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


#matplotlib parameters
matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size':'25', 'xtick.labelsize':'20', 'ytick.labelsize':'20'})

#retrieve and sort topofiles
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()

for i in range(len(topofiles)):
    data = open("{}_Pert.csv".format(topofiles[i]))
    data = data.readlines()
    for j in range(len(data)):
        data[j]  = data[j].split(',')
    network_name = topofiles[i]
    
    node_names = []
    node_id = []
    node_id_file = open("input/{}.ids".format(topofiles[i])).read().split("\n")[1:]
    if "" in node_id_file:
            node_id_file.remove("")    
    for u in node_id_file:
        temp = u.split(" ")
        node_names.append(temp[0])
        node_id.append(int(temp[1]))
        
    #creating a node to id dictionary
    id_dict = dict(zip(node_names, node_id))
    import matplotlib as mpl
    
    cmap = plt.get_cmap('gray')    
    new_cmap = truncate_colormap(cmap, 0, 0.6)
    mpl.rc('image', cmap= new_cmap)

    print(id_dict)
    link_matrix = metric.matrix(topofiles[i])
    copy_link_matrix = link_matrix.copy()
    


    graph = nx.DiGraph()
    n = len(link_matrix)
        
    for p in range(n):
            for q in range(n):
                if(link_matrix[p][q] != 0):
                    graph.add_edge(p,q,weight =link_matrix[p][q]  )    
    #info about + and - cycles               
    cycle_info = metric.cycle_info(topofiles[i] , graph)
    
    l = len(data)  - 1
   
    orig_pos = np.array([cycle_info[0]]*l)
    orig_neg = np.array([cycle_info[1]]*l)
    orig_fwpc = np.array([cycle_info[3]]*l)
    
    orig_weight_pos = np.array([cycle_info[5]]*l)
    orig_weight_neg = np.array([cycle_info[4]] *l  )
        
    new_pos =[]
    new_fwpc =[]
    new_weight_pos = []
    new_weight_neg = []
    plastarr = []
    colorarr = []    
    

    for j in range(1, len(data) ):
        n1 = 0
        n2 = 0
        str1 = data[j][0][:-4]
        print(data[j][0])
        newval = int(data[j][0][len(data[j][0]) -1])
        
        #reading the csv file
        for k in range(len(str1)):
            if(str1[k]=='-'):   
                try:
                    n1 = id_dict[str1[0:k]]        
                    n2 = id_dict[str1[k+1:len(str1)]]
                    
                    if(newval == 2):
                        newval = -1         
                    link_matrix[n1][n2] = newval
                    break
                except:
                    pass
                    
        #plasticity and generating link_matrix                    
        plastval = float(data[j][4])    
        graph = nx.DiGraph()
        n = len(link_matrix)
        
        for p in range(n):
            for q in range(n):
                if(link_matrix[p][q] != 0):
                    graph.add_edge(p,q,weight =link_matrix[p][q]  )
        
        print(j)
        
        network_name = topofiles[i]
        cycle_info = metric.cycle_info(network_name, graph)
        

        colorarr.append(cycle_info[1])
        new_pos.append(cycle_info[0])
        
        new_fwpc.append(cycle_info[3])
        
        plastarr.append(plastval)
        print(cycle_info[0] , cycle_info[3] , plastval)
        link_matrix[n1][n2] = copy_link_matrix[n1][n2]
    
    
    new_pos1 = new_pos - orig_pos    
    new_fwpc1 = new_fwpc - orig_fwpc 
    #optimising weights
    #temp_frac_weight_loops , popt_frac = fit_frac(new_weight_pos,new_weight_neg, plastarr)

    

    #plotting vs diff in positive feedback loops 
    r = 2
    fig = plt.figure()
    if(network_name== "EMT_RACIPE"):
        plt.scatter(new_pos1 , plastarr , c= colorarr, s = np.power((np.array(colorarr) + 300)/10 , 1.2 ))
    else:
        plt.scatter(new_pos1 , plastarr , c= colorarr, s = np.power((np.array(colorarr) + 2100)/70 , 1.2 ))
    
    
    plt.colorbar()
    
    corr, significance= pearsonr(new_pos1, plastarr)
    
    plt.xlabel("Δ PFLs")
    plt.ylabel("Perturbed Plasticity")
    plt.title("{}  ρ = {:.3f}{}".format(network_name,corr, starfunc(significance)))
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)          

    
    plt.savefig("just_pos_{}.png".format(topofiles[i]) , transparent =True)
    plt.clf()
    
    #plotting vs diff in fwpc
    r = 2
    fig = plt.figure()
    plt.scatter(new_fwpc1, plastarr , c= colorarr, s = 70)
    plt.colorbar()   
    corr, significance = pearsonr(new_fwpc1, plastarr)    
    
    plt.xlabel("Δ FWPC")
    plt.ylabel("Perturbed Plasticity")
    plt.title("{}  ρ = {:.3f}{}".format(network_name,corr, starfunc(significance)))
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)          
     
    plt.savefig("diff_weight_frac_{}.png".format(topofiles[i]) , transparent =True)
    plt.clf()     
    

    