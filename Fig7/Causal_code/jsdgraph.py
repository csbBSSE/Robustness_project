import numpy as np
import matplotlib.pyplot as plt
from math import pow
import os
from scipy.stats import norm, zscore
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from matplotlib import rcParams

ising=open("ising_jsd.txt")
cont=open("cont_jsd.txt")


ising_jsddata = ising.read().split("\n")[0:]

if "" in ising_jsddata:
    ising_jsddata.remove("")
    
ising.close()


cont_jsddata = cont.read().split("\n")[0:]

if "" in cont_jsddata:
    cont_jsddata.remove("")
    
cont.close()
contarr=[0]*len(cont_jsddata)
isingarr=[0]*len(cont_jsddata)

contbio=[]
isingbio=[]


col=['b']*len(cont_jsddata)
u=0

for i in ising_jsddata :
    temp=i.split(" ")
    isingarr[u]=float(temp[1])
    u+=1
    if(temp[0][0]=='r'):
        col[u]='r'
    if(temp[0][0]!='r' and temp[0][0]!='a'):
        isingbio.append(float(temp[1]))
u=0

for i in cont_jsddata :
    temp=i.split(" ")
    contarr[u]=float(temp[1])    
    if(temp[0][0]!='r' and temp[0][0]!='a'):
        contbio.append(float(temp[1]))
    u+=1
fig, ax = plt.subplots(1,1)    
plt.scatter(contarr,isingarr,color=col,s=5)   
x = np.linspace(0, 1, 100)
plt.plot(x,x,color='g')
plt.title("Ising_avgJSD= {}     Cont_avgJSD={}".format( np.round(np.mean(isingarr),4)  ,np.round(np.mean(contarr),4) ) )
ax.set_xlabel("Cont JSD from RACIPE", color='r')
ax.set_ylabel("Ising JSD from RACIPE", color='r')
plt.tight_layout()
plt.savefig("contvsisingjsd1.jpg")



plt.clf()
contbio=np.array(contbio)
isingbio=np.array(isingbio)
plt.plot(x,x,color='g')
plt.scatter(contbio,isingbio,s=5)
plt.title("Isingbio_avgJSD= {}     Contbio_avgJSD={}".format( np.round(np.mean(isingbio),4)  ,np.round(np.mean(contbio),4) ) )
plt.savefig("contvsisingjsdbio.jpg")