import os
import time
from os import listdir
from os.path import splitext, isfile, join
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
from copy import copy
from scipy.optimize import curve_fit
print('imported modules')


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
                z[i] = (a*x[i])/(a*x[i] + y[i])
            except:
                z[i] = 0
    return b*z + z1

def fit_frac(x_arr,y_arr, target_arr):
  p0 = 0.3 , 1.0 , 1.0
  y_arr = np.array(y_arr)
  x_arr = np.array(x_arr)
  
  popt, pcov = curve_fit(func_frac, (x_arr,y_arr), target_arr, p0)
  popt = np.array(popt)

  z_arr = func_frac_output( (x_arr, y_arr) , popt[0], popt[1] , popt[2])
  return z_arr , popt
  
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
    mpl.rc('image', cmap='winter')
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
    orig_weight_pos = np.array([cycle_info[5]]*l)
    orig_weight_neg = np.array([cycle_info[4]] *l  )
    orig_frac_weight  = []
    for k in range(len(orig_pos)):
        orig_frac_weight.append(3*orig_weight_pos[k]/(orig_weight_neg[k] + 3* orig_weight_pos[k]))        
        
    new_pos =[]
    new_neg =[]
    new_weight_pos = []
    new_weight_neg = []
    plastarr = []


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
        

  
        new_pos.append(cycle_info[0])
        new_neg.append(cycle_info[1])
        new_weight_pos.append(cycle_info[5])
        new_weight_neg.append(cycle_info[4])
        
        plastarr.append(plastval)
        print(cycle_info[0] , cycle_info[5] , plastval)
        link_matrix[n1][n2] = copy_link_matrix[n1][n2]
    
    
    new_pos1 = new_pos - orig_pos    
    new_neg1 = new_neg - orig_neg
    
    #optimising weights
    temp_diff_weight_loops , popt_diff = fit_diff(new_weight_pos,new_weight_neg, plastarr)
    temp_frac_weight_loops , popt_frac = fit_frac(new_weight_pos,new_weight_neg, plastarr)
    
    print("dfsfsfsfssfsf s ")
    print(popt_frac)
    print(func_frac_output( (new_weight_pos, new_weight_neg) , popt_frac[0], popt_frac[1] , popt_frac[2]))
    
    print("positive and neg loops")
    for qq in range (len(new_weight_neg)):
        print(new_weight_pos[qq] , new_weight_neg[qq] , new_pos[qq] , new_neg[qq] , (popt_frac[0]*new_weight_pos[qq])/(popt_frac[0]*new_weight_pos[qq] + new_weight_neg[qq] ))
        
    diff_weight_loops = temp_diff_weight_loops - func_diff_output( (orig_weight_pos, orig_weight_neg) , popt_diff[0], popt_diff[1] , popt_diff[2])



    #plotting vs diff in positive feedback loops 
    r = 2
    fig = plt.figure()
    plt.scatter(new_pos1 , plastarr , c= new_neg, s = 50)
    plt.colorbar()
    
    corr, _ = pearsonr(new_pos1, plastarr)
    
    plt.xlabel("Δ Positive Feedback Loops")
    plt.ylabel("Plasticity of Perturbed Network")
    plt.title("{}  ρ = {:.3f}".format(network_name,corr))
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)          

    
    plt.savefig("just_pos_{}.jpg".format(topofiles[i]))
    plt.clf()
    
    
    #plotting vs diff in negative feedback loops 
    r = 2
    fig = plt.figure()
    plt.scatter(new_neg1 , plastarr , c= new_pos, s = 50)
    plt.colorbar()
    
    corr, _ = pearsonr(new_neg1, plastarr)
    
    plt.xlabel("Δ Negative Feedback Loops")
    plt.ylabel("Plasticity of Perturbed Network")
    plt.title("{}  ρ = {:.3f}".format(network_name,corr))
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)          

    
    plt.savefig("just_neg_{}.jpg".format(topofiles[i]))
    plt.clf()    
    
    
    #plotting vs diff in weighted feedback loops    
    r = 2
    fig = plt.figure()
    plt.scatter(diff_weight_loops, plastarr , c= new_neg , s = 50)
    plt.colorbar()   
    corr, _ = pearsonr(diff_weight_loops, plastarr)    
    
    plt.xlabel("Δ Weighted Feedback Loops")
    plt.ylabel("Plasticity of Perturbed Network")   
    plt.title("{}  ρ = {:.3f}".format(network_name,corr))
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)          
     
    plt.savefig("diff_weight_loops_{}.jpg".format(topofiles[i]))
    plt.clf()     
  
    frac_weight = []
    
     #plotting vs diff in frac weighted feedback loops    
    for k in range(len(new_weight_pos)):
        frac_weight.append(3* new_weight_pos[k]/(new_weight_neg[k] + 3* new_weight_pos[k]))
    frac_weight = np.array(frac_weight) 
    frac_weight1 =  frac_weight - orig_frac_weight 
     
    r = 2
    fig = plt.figure()
   
    plt.scatter(frac_weight1, plastarr , c= new_neg, s = 50)
    plt.colorbar()   
    corr, _ = pearsonr(frac_weight1, plastarr)    
    
    plt.xlabel("Δ Fraction of Weighted Feedback Loops",fontweight="bold" , c='0.3')
    plt.ylabel("Plasticity of Perturbed Network",fontweight="bold" , c='0.3')   
    plt.title("{}  ρ = {:.3f}".format(network_name,corr),fontweight="bold" , c='0.3' )
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)         
    
    plt.savefig("weight_combinedfrac_{}.jpg".format(topofiles[i]))
    plt.clf()  

     #plotting vs diff in frac weighted feedback loops    

    print(temp_frac_weight_loops)
    r = 2
    fig = plt.figure()
   
    plt.scatter(temp_frac_weight_loops, plastarr , c= new_neg, s = 50)
    plt.colorbar()   
    corr, _ = pearsonr(temp_frac_weight_loops, plastarr)    
    
    plt.xlabel("Δ Fraction of Weighted Feedback Loops",fontweight="bold" , c='0.3')
    plt.ylabel("Plasticity of Perturbed Network",fontweight="bold" , c='0.3')   
    plt.title("{}  ρ = {:.3f}".format(network_name,corr),fontweight="bold" , c='0.3' )
    f=r*np.array(plt.rcParams["figure.figsize"])

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)         
    plt.xticks(fontsize=5)    
    plt.savefig("weight_combinedfractemp_{}.jpg".format(topofiles[i]))
    plt.clf()       


    