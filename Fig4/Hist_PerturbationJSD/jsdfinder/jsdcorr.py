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
import os
import time
from os import listdir
from os.path import isfile, join



topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]


for w in range(len(topofiles)):
    #arr1=[0,0.0224,0,0.4718,0.4793,0,0.0265,0]
    fig, axs = plt.subplots(7)
    #plt.figure(figsize=(10,5))
    rcParams.update({'figure.autolayout':True})
    #probjsd1 = np.loadtxt("GRHL2_async_unweigh_jsd_{}.txt".format(1))
    #arr1= probjsd1[99,:]



    data = pd.read_csv("{}_JSD.txt".format(topofiles[w]) , header = None,sep = " ")
    length=len(data.columns)-1
    wtarr=[0]*length
    perturbarr=[[0]*length]*(len(data.index) -1)
    jsdarr=[0]*(len(data.index) -1)
    
    for q in range(1,length+1):
        wtarr[q-1]=data[0:1][q]

    for p in range(1,len(data.index)):
        for q in range(1,length+1):
            perturbarr[p-1][q-1]=data[p:p+1][q]
     
        jsdarr[p-1]=jensenshannon(wtarr,perturbarr[p-1],2)


    jsdfile=open("cnt2_{}_jsd.txt".format(topofiles[w]),'w')

    for i in range(len(jsdarr)):
        if(i==len(jsdarr) -1) :
            jsdfile.write("{:.6f}".format(float(jsdarr[i])))
        else:
            jsdfile.write("{:.6f}\n".format(float(jsdarr[i])))


    print(jsdarr)


    print(jsdarr)



