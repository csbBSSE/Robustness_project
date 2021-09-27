import numpy as np
import matplotlib.pyplot as plt
from math import pow
import os
from scipy.stats import norm, zscore
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from matplotlib import rcParams
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
import matplotlib
import seaborn as sns
from scipy.stats.stats import pearsonr

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '40', 'axes.titlesize':'40', 'axes.labelsize':'35', 'xtick.labelsize':'33', 'ytick.labelsize':'30', 'legend.fontsize':'40'})

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]

topofiles.sort()
print(topofiles)

version='bool' # bool / cont
Version = version

def FixCase(st):
    return ' '.join(''.join([w[0].upper(), w[1:].lower()]) for w in st.split())

corrarr= []

def starfunc(significance):
    if significance < 0.001:
        return "***"
    elif significance < 0.01:
        return "**"
    elif significance < 0.05:
        return "*"
    else:
        return ""

for i in range(0, len(topofiles)):
    
    version1= version
    if(version == 'cont'):
        version = 'cnt2'
    if(version == 'bool'):
        version = 'ising'
    
    racjsd=np.loadtxt("data/racipe_{}_jsd.txt".format(topofiles[i]))
    booljsd=np.loadtxt("data/{}_{}_jsd.txt".format(version ,topofiles[i] ))
 
    
    r = 2
    fig,ax = plt.subplots()
    matplotlib.rcParams.update({'font.size': 10*r})
    corr, significance = pearsonr(racjsd,booljsd)
    print(topofiles[i], significance)
    corrarr.append(corr)
    a,res, rank , sing, thres = np.polyfit(booljsd,racjsd, 1, full = True)
    m = a[0]
    b = a[1]
    
    seagraph = sns.regplot(booljsd,racjsd, color ='blue')
    
    xlabel1 = "Perturbation JSD ({})".format(FixCase(Version))
    ylabel1= "Perturbation JSD (RACIPE)"
    #title1 = "{}   ρ= {:.3f}   Residual = {:.3f} ".format(topofiles[i],corr, res[0])
    if(topofiles[i] != 'OVOL2'):
        title1 = "{}".format(topofiles[i])
    else:
        title1 = "{}".format('OVOL')
    textstr = "ρ= {:.3f}{} \nResidual = {:.3f} ".format(corr, starfunc(significance), res[0])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize = 40,
        verticalalignment='top', bbox=props)
    
    
    
    seagraph.set(xlabel = xlabel1, ylabel = ylabel1, title = title1)
    plt.plot(booljsd, m*booljsd + b, c='r')    
    plt.title(title1, x = 0.9, y = 0.9)
    figure = seagraph.get_figure()  
    lim1 = min(booljsd) - 0.05
    lim2 = max(booljsd) + 0.05


    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)   
    plt.tight_layout()       
    plt.autoscale()
    ax.set(xlim=(lim1 , lim2))
    figure.savefig("{}plots/jsd_{}_seaborn.png".format(Version,topofiles[i]) , transparent = True)
    plt.clf()


corrarr = np.array(corrarr).T
np.set_printoptions(precision=None, suppress=None) 
np.savetxt("{}_corr.txt".format(version) , corrarr)







