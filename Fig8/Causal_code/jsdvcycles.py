import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math
import initialise.initialise as initialise
import initialise.parser as parser
import os
import time
from os import listdir
from os.path import isfile, join
from scipy import stats

def polyfit(x, y, degree):
    results = {}

    coeffs = numpy.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = numpy.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = numpy.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = numpy.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = numpy.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results
    
    


topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]


cycledata= np.loadtxt("networkCycles.txt" , skiprows =1 , usecols=range(1,9) )
bioindex=[]

cycledata=np.matrix(cycledata)
cycledata= cycledata.T

cont=open("cont_jsd.txt")

cont_jsddata = cont.read().split("\n")[0:]

if "" in cont_jsddata:
    cont_jsddata.remove("")
    
cont.close()

contarr=[0]*len(cont_jsddata)

u=0

for i in cont_jsddata :
    temp=i.split(" ")
    contarr[u]=float(temp[1])    
    str1=temp[0]
    if(str1[0]!='r' and str1[0]!='a'):
        bioindex.append(u)
    u+=1
#print(cycledata[2].flatten())



neg_arr= np.array( cycledata[1] ).flatten()
neg_weight_arr= np.array( cycledata[5] ).flatten()
neg_fracarr= np.array( cycledata[2] ).flatten()
neg_weight_fracarr= np.array( cycledata[6] ).flatten()


pos_arr= np.array( cycledata[1] ).flatten()
pos_weight_arr= np.array( cycledata[5] ).flatten()
pos_fracarr= np.array( cycledata[2] ).flatten()
pos_weight_fracarr= np.array( cycledata[6] ).flatten()
#print(neg_arr)
for q in range (len(neg_arr)):
    #neg_arr[q]=neg_arr [q] /(neg_arr [q]+1)
    #neg_arr[q]= min ( neg_arr[q],50 )

    if(neg_arr[q]>50):
        neg_arr[q]=50


for q in range (len(neg_arr)):
    #neg_arr[q]=neg_arr [q] /(neg_arr [q]+1)
    #neg_arr[q]= min ( neg_arr[q],50 )

    if(neg_weight_arr[q]>50):
        neg_weight_arr[q]=15
        


plt.scatter(neg_fracarr,contarr, s= 10)
plt.savefig("negcyclevsjsdfracarr.jpg")
plt.clf()



fig, ax = plt.subplots(1,1)
plt.scatter(neg_weight_fracarr,contarr, s= 10)
neg_weight_fracarr=np.array(neg_weight_fracarr)
contarr=np.array(contarr)
gradient, intercept, r_value, p_value, std_err = stats.linregress(neg_weight_fracarr,contarr)
mn=np.min(neg_weight_fracarr)
mx=np.max(neg_weight_fracarr)
x1=np.linspace(mn,mx,500)
y1=gradient*x1+intercept
plt.plot(x1,y1,'-r')

plt.title("JSD b/w Cont and  RACIPE vs Cycle measure        R^2 = {:.4f}".format (r_value**2) , color= 'r')
ax.set_xlabel("Fraction of Negative weighted Cycles", color='r')
ax.set_ylabel("JSD b/w Cont and  RACIPE", color='r')




plt.tight_layout()


plt.savefig("negcyclevsjsdneg_weight_fracarr1.jpg")
plt.clf()

plt.scatter(neg_arr,contarr, s= 10)
plt.savefig("negcyclevsjsdneg_arr.jpg")
plt.clf()

plt.scatter(neg_weight_arr,contarr, s= 10)



plt.savefig("negcyclevsjsdneg_weight_arr1.jpg")
plt.clf()



bioneg_arr = [neg_arr[i] for i in bioindex]
bioneg_weight_arr= [neg_weight_arr[i] for i in bioindex]
bioneg_fracarr= [neg_fracarr[i] for i in bioindex]
bioneg_weight_fracarr= [neg_weight_fracarr[i] for i in bioindex]
biocontarr= [contarr[i] for i in bioindex]
bionets= [topofiles[i] for i in bioindex]


for q in range(len(biocontarr)):
    print(bionets[q],biocontarr[q],bioneg_arr[q])


plt.scatter(bioneg_fracarr,biocontarr, s= 10)
plt.savefig("bionegcyclevsjsdfracarr.jpg")
plt.clf()

plt.scatter(bioneg_weight_fracarr,biocontarr, s= 10)
plt.savefig("bionegcyclevsjsdneg_weight_fracarr.jpg")
plt.clf()

plt.scatter(bioneg_arr,biocontarr, s= 10)
plt.savefig("bionegcyclevsjsdneg_arr.jpg")
plt.clf()

plt.scatter(bioneg_weight_arr,biocontarr, s= 10)
plt.savefig("bionegcyclevsjsdneg_weight_arr.jpg")
plt.clf()
