import os
import time
from os import listdir
from os.path import isfile, join
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()
import matplotlib
import numpy as np
import sys
import initialise.initialise as initialise
import initialise.parser as parser
import modules.metric as metric
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '30', 'axes.titlesize':'30', 'axes.labelsize':'30', 'xtick.labelsize':'25', 'ytick.labelsize':'25', 'legend.fontsize':'25'})
plt.rcParams['figure.dpi'] = 500
line_arr = ['solid' , 'dashed', 'dashdot', 'dotted']

def file_read(fname):
    arr1  = []
    file = open(fname, 'r')
    Lines = file.readlines()
    for i in Lines:
        arr1.append(float(i))
    file.close()
    return arr1

def namestrip(net_name):
    if net_name.endswith('_fix'):
     return net_name[:-4]
    else:
        return net_name
    
    
    
def barplotter(net_name):

    fig,ax = plt.subplots()
    dn_fwpc = file_read("Run1/{}_dn_fraccycles.txt".format(net_name))
    up_fwpc = file_read("Run1/{}_up_fraccycles.txt".format(net_name))
    
    x_arr = [i for i in range(len(dn_fwpc))]
    x_arr_str = [str(i) for i in range(len(dn_fwpc))]
    plt.xticks(x_arr,x_arr_str)
    x_arr = np.array(x_arr)
    plt.bar(x_arr - 0.2, dn_fwpc, width = 0.4 ,color='g' , hatch = 'o' , edgecolor = 'k', linewidth= 2)
    
    x_arr = [i for i in range(len(up_fwpc))]
    x_arr_str = [str(i) for i in range(len(up_fwpc))]
    plt.xticks(x_arr,x_arr_str)
    x_arr = np.array(x_arr)
    plt.bar(x_arr + 0.2, up_fwpc, width = 0.4 ,color='b' , hatch = 'x' , edgecolor = 'k', linewidth= 2)
    plt.legend(['Increase FWPC', 'Decrease FWPC'])
    plt.title("{}".format(namestrip(net_name)))
    r = 2
    f=r*np.array(plt.rcParams["figure.figsize"])

    plt.xlabel("No. of Perturbations", fontweight="bold", c = '0.3')
    plt.ylabel("FWPC", fontweight="bold", c = '0.3')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)            
    plt.tight_layout()
    plt.savefig("{}_barplot.png".format(namestrip(net_name)), transparent = True)
    plt.clf()
    plt.close(fig)

def lineplotter(net_name):
    fig,ax = plt.subplots()
    
    dn_pjsd = [ [], [], [] ]
    up_pjsd = [ [], [], [] ]
    
    for i in range(1,4):
        dn_pjsd[i-1] = file_read("Run{}/{}_dn_jsd.txt".format(i,net_name))
        up_pjsd[i-1] = file_read("Run{}/{}_up_jsd.txt".format(i,net_name))   

    
    
    dn_pjsd_mean  = []
    dn_pjsd_err = []
    up_pjsd_mean = []
    up_pjsd_err =[]
    data_arr = dn_pjsd
    
    for i in range(len(dn_pjsd[0])):

        dn_pjsd_mean.append(np.mean([data_arr[0][i], data_arr[1][i] , data_arr[2][i]])) 
        dn_pjsd_err.append(np.std([data_arr[0][i], data_arr[1][i] , data_arr[2][i]]))
        
    data_arr = up_pjsd
    
    for i in range(len(up_pjsd[0])):
        up_pjsd_mean.append(np.mean([data_arr[0][i], data_arr[1][i] , data_arr[2][i]])) 
        up_pjsd_err.append(np.std([data_arr[0][i], data_arr[1][i] , data_arr[2][i]]))
    
    x_arr = [i for i in range(len(dn_pjsd_mean))]
    x_arr_str = [str(i) for i in range(len(dn_pjsd_mean))]
    plt.xticks(x_arr,x_arr_str)
    x_arr = np.array(x_arr)
    #plt.plot(x_arr, dn_pjsd_mean, color='b' , linestyle = 'solid' , linewidth = 4)
    #plt.errorbar(x_arr,dn_pjsd_mean, dn_pjsd_err, ecolor='k', elinewidth = 4 , capsize = 12)
    plt.errorbar(x_arr, dn_pjsd_mean, dn_pjsd_err, color='b' , linestyle = 'solid' , linewidth = 4, ecolor='k', elinewidth = 4, capsize = 16)
    
    x_arr = [i for i in range(len(up_pjsd_mean))]
    x_arr_str = [str(i) for i in range(len(up_pjsd_mean))]
    plt.xticks(x_arr,x_arr_str)
    x_arr = np.array(x_arr)
    plt.errorbar(x_arr, up_pjsd_mean,up_pjsd_err, color='g' , linestyle = 'dashed' , linewidth = 4, ecolor='k', elinewidth = 4, capsize = 16)

    
    plt.legend(['Increase FWPC', 'Decrease FWPC'])
    plt.title("{}".format(namestrip(net_name)))
    r = 2
    f=r*np.array(plt.rcParams["figure.figsize"])

    plt.xlabel("No. of Perturbations", fontweight="bold", c = '0.3')
    plt.ylabel("PJSD", fontweight="bold", c = '0.3')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)            
    plt.tight_layout()
    plt.savefig("{}_lineplot.png".format(namestrip(net_name)), transparent = True)
    plt.clf()    
    plt.close(fig)
    
    
def lineplotter2(fig,ax,net_name, c1,lstyle):
    dn_pjsd = [ [], [], [] ]
    up_pjsd = [ [], [], [] ]
    
    for i in range(1,4):
        dn_pjsd[i-1] = file_read("Run{}/{}_dn_jsd.txt".format(i,net_name))
        up_pjsd[i-1] = file_read("Run{}/{}_up_jsd.txt".format(i,net_name))   

    dn_pjsd_mean  = []
    dn_pjsd_err = []
    up_pjsd_mean = []
    up_pjsd_err =[]
    data_arr = dn_pjsd
    
    for i in range(len(dn_pjsd[0])):

        dn_pjsd_mean.append(np.mean([data_arr[0][i], data_arr[1][i] , data_arr[2][i]])) 
        dn_pjsd_err.append(np.std([data_arr[0][i], data_arr[1][i] , data_arr[2][i]]))
        
    data_arr = up_pjsd
    
    for i in range(len(up_pjsd[0])):
        up_pjsd_mean.append(np.mean([data_arr[0][i], data_arr[1][i] , data_arr[2][i]])) 
        up_pjsd_err.append(np.std([data_arr[0][i], data_arr[1][i] , data_arr[2][i]]))
    
    x_arr = [i for i in range(len(dn_pjsd_mean))]
    x_arr_str = [str(i) for i in range(len(dn_pjsd_mean))]
    plt.xticks(x_arr,x_arr_str)
    x_arr = np.array(x_arr)
    #plt.plot(x_arr, dn_pjsd_mean, color='b' , linestyle = 'solid' , linewidth = 4)
    #plt.errorbar(x_arr,dn_pjsd_mean, dn_pjsd_err, ecolor='k', elinewidth = 4 , capsize = 12)
    plt.errorbar(x_arr, dn_pjsd_mean, dn_pjsd_err, color=c1 , linestyle = lstyle , linewidth = 4, ecolor='k', elinewidth = 4, capsize = 16)
    
    x_arr = [i for i in range(len(up_pjsd_mean))]
    x_arr_str = [str(i) for i in range(len(up_pjsd_mean))]
    plt.xticks(x_arr,x_arr_str)
    x_arr = np.array(x_arr)
    plt.errorbar(x_arr, up_pjsd_mean, up_pjsd_err, color=c1 , linestyle = lstyle , linewidth = 4, ecolor='k', elinewidth = 4, capsize = 16, label='_nolegend_')
    
    
def lineplotter3(fig,ax,net_name, c1,lstyle):
    dn_fwpc= file_read("Run1/{}_dn_fraccycles.txt".format(net_name))
    up_fwpc = file_read("Run1/{}_up_fraccycles.txt".format(net_name))
    

    
    x_arr = [i for i in range(len(dn_fwpc))]
    x_arr_str = [str(i) for i in range(len(dn_fwpc))]
    plt.xticks(x_arr,x_arr_str)
    x_arr = np.array(x_arr)
    plt.plot(x_arr, dn_fwpc, color=c1 , linestyle = lstyle , linewidth = 4)

    x_arr = [i for i in range(len(up_fwpc))]
    x_arr_str = [str(i) for i in range(len(up_fwpc))]
    plt.xticks(x_arr,x_arr_str)
    x_arr = np.array(x_arr)
    plt.plot(x_arr, up_fwpc, color=c1 , linestyle = lstyle , linewidth = 4,  label='_nolegend_')   
    
    

fig,ax = plt.subplots()    
cols = ['r', 'b', 'g']
styles = ['solid', 'dashed', 'dotted']
for i in range(2,5):
    lineplotter3(fig, ax, "randomnet8_{}".format(i) + "_fix", cols[i-2] , styles[i-2])

plt.legend( ['randomnet8_2'  , 'randomnet8_3'  ,'randomnet8_4'     ])
r = 2
    
f=r*np.array(plt.rcParams["figure.figsize"])

plt.xlabel("No. of Perturbations", fontweight="bold", c = '0.3')
plt.ylabel("FWPC", fontweight="bold", c = '0.3')
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)            
plt.tight_layout()
plt.savefig("lineplotFWPC_combined.png", transparent = True)
plt.clf()    
plt.close(fig)     
    



 


fig,ax = plt.subplots()    
cols = ['r', 'b', 'g']
styles = ['solid', 'dashed', 'dotted']
for i in range(2,5):
    
    lineplotter2(fig, ax, "randomnet8_{}".format(i) + "_fix", cols[i-2] , styles[i-2])

plt.legend( ['randomnet8_2'  , 'randomnet8_3'  ,'randomnet8_4'     ])
r = 2
    
f=r*np.array(plt.rcParams["figure.figsize"])

plt.xlabel("No. of Perturbations", fontweight="bold", c = '0.3')
plt.ylabel("PJSD", fontweight="bold", c = '0.3')
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)            
plt.tight_layout()
plt.savefig("lineplotPJSD_combined.png", transparent = True)
plt.clf()    
plt.close(fig)     
    



for i in range(len(topofiles)):   
    lineplotter(topofiles[i])
    barplotter(topofiles[i])




   
   
   
   