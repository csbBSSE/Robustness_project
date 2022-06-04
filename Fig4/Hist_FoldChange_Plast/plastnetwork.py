import os
import time
from os import listdir
from os.path import isfile, join
import numpy as np

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles4/") if isfile(join("topofiles4/", f))]
topofiles.sort()

print(topofiles)
datafile=open("plastnetwork.txt",'w+')

for i in range(len(topofiles)):
    
    plast=np.loadtxt("data/{}_plastdata.txt".format(topofiles[i]))
    
    plast1=[]
    div=plast[0]
    if(div<0.0001):
        div=0.0001
    for k in range(1,len(plast)):
        if(plast[k]/div <1):
            plast1.append(plast[k]/div)
        else:
            plast1.append(div/plast[k])
     
    datafile.write("{} {}\n".format(np.mean(plast1))

    