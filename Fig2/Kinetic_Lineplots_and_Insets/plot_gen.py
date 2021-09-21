import os
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()
matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '50', 'axes.titlesize':'50', 'axes.labelsize':'50', 'xtick.labelsize':'40', 'ytick.labelsize':'40'})
nn = "GRHL2"

def fc(a,b):
    return min(a,b)


amplifications = [1, 2, 3, 0.33, 0.5]
amplifications.sort()

colours = ["r", "g", "b"]
legendarr = ["All Parameters", "No Hill", "Only Hill"]

jsdbar = [[],[]]
plastbar = [[],[]]
for nn in topofiles:
    jsd = np.loadtxt("ALL/{}_JSD.txt".format(nn))
    plast = np.loadtxt("ALL/{}_PLAST.txt".format(nn))

    one3 = jsd[4,:]
    p51 = jsd[0,:]
    avg = [(jsd[0,0] + jsd[4,0])/2, (jsd[0,4] + jsd[4,4])/2]
    jsdbar[0].append(avg[0])
    jsdbar[1].append(avg[1])

    one3 = plast[2,:] / plast[4,:]
    p51 = plast[0,:] / plast[2,:]	
    avg = [(plast[0,0]/plast[2,0] + plast[2,0]/plast[4,0])/2, (plast[0,4]/plast[2,4] + plast[2,4]/plast[4,4])/2]
    plastbar[0].append(avg[0])
    plastbar[1].append(avg[1])

xarrno = np.linspace(0, len(jsdbar[0]), len(jsdbar[0]))

r = 2
fig = plt.figure()

#matplotlib.rcParams.update({'font.size': 10*(r+0.5)})
plt.rc('legend',fontsize=10*(r+0.5))
plt.bar(xarrno, jsdbar[0], width = 0.4 ,color='r')
plt.xticks(xarrno + 0.2, topofiles, fontweight="bold" , c='0.3' )
plt.bar(xarrno + 0.4, jsdbar[1], width = 0.4, color='b')
plt.xticks(rotation=20)
#plt.xticks(fontsize= 10*(r+0.5))
plt.ylabel("Average JSD", fontweight="bold" , c='0.3' )
plt.xlabel("Networks" ,fontweight="bold" , c='0.3' )


legend = plt.legend(["All Parameters Varied", "Only Hill Coeff Varied"] )
plt.setp(legend.get_texts(), color='0.3',fontsize = 10*(r+0.5) , fontweight="bold" )

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off

f=(r + 0.7)*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)
plt.title("Average JSD", fontweight="bold")
axes = plt.gca()
axes.set_ylim([0,0.27])
plt.tight_layout()
plt.savefig("barjsd.png", transparent = True)
plt.close()

r = 2
fig = plt.figure()
aaa = 14
#matplotlib.rcParams.update({'font.size': 12*(r+0.5)})
plt.rc('legend',fontsize=aaa*(r+0.5))
plt.bar(xarrno, plastbar[0], width = 0.4 ,color='r')
plt.xticks(xarrno + 0.2 , topofiles, fontweight="bold" , c='0.3' )
plt.bar(xarrno+0.4, plastbar[1], width = 0.4, color='b')
plt.xticks(rotation=20)
#plt.xticks(fontsize= aaa*(r+0.5))
plt.ylabel("Average Fold Change", fontweight="bold" , c='0.3' )
plt.xlabel("Networks", fontweight="bold" , c='0.3' )


legend = plt.legend(["All Parameters Varied", "Only Hill Coeff Varied"] )
plt.setp(legend.get_texts(), color='0.3',fontsize = aaa*(r+0.5) , fontweight="bold" )

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off
plt.title("Average Fold Change", fontweight="bold")
f=(r + 0.7)*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)
axes = plt.gca()
axes.set_ylim([0,1])
plt.tight_layout()
plt.savefig("barplast.png", transparent = True)
