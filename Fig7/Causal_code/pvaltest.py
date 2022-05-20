import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
import os
import time
from os import listdir
from os.path import isfile, join


cont4=open("cont4_jsd.txt")

cont4_jsddata = cont4.read().split("\n")[0:]

if "" in cont4_jsddata:
    cont4_jsddata.remove("")
    
cont4.close()

cont4arr=[0]*len(cont4_jsddata)

u=0

for i in cont4_jsddata :
    temp=i.split(" ")
    cont4arr[u]=float(temp[1])    
    str1=temp[0]
    
    #if(str1[0]!='r' and str1[0]!='a'):
    #    bioindex.append(u)
    #u+=1
    
xarr= np.linspace(0,1,25)

plt.histogram(xarr,cont4arr)
plt.savefig("jsd4hist.jpg")    
    
    
    