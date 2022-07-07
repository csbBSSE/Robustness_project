import os
import time
from os import listdir
from os.path import isfile, join
from os.path import splitext, isfile, join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
print('imported modules')
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()


matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '25', 'axes.titlesize':'20', 'axes.labelsize':'20', 'xtick.labelsize':'20', 'ytick.labelsize':'15', 'legend.fontsize':'15'})


colorarr = []
c = ['r' , 'b', 'g', 'm']
labels = ['100 ODEs' , '1000 ODEs' , '10000 ODEs']

for i in range(3):
    colorarr.append( mpatches.Patch(color=c[i], label=labels[i]) )

for i in range(len(topofiles)):
  runs = [100,1000,10000]
  cnt = -1
  colarr = ['r', 'b', 'g', 'm']
  for j in runs:
    toponame = "{}_{}".format(topofiles[i], j)
    
    arr = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    network = topofiles[i]
    data = np.loadtxt("{}_parameters.dat".format(toponame))
    
    length = len(data)

    for j in range(0,3):
        for k in range(j*(length//3) , (j+1)*(length//3) ):
            if(int(data[k][1]) >=4):
                arr[j][3]+=1
            else:
                arr[j][int(data[k][1]) -1] +=1
    meanarr = (arr[0] + arr[1] + arr[2])/3

    scaling = np.sum(meanarr)
    meanarr = meanarr/scaling
    arr = arr/scaling
    stdarr = [0,0,0,0]
    print(arr)
    print(arr[0][1])
    for j in range(4):
       std = np.std([arr[0][j], arr[1][j] , arr[2][j]] )
       stdarr[j] = std
    

    
    arr2 = [1,2,3,4]
    arr3 = np.array(arr2)
    arr3 = arr3 + cnt*np.array([0.25,0.25,0.25,0.25])
    print(arr3)
    #plt.bar(arr3, meanarr, color=['r', 'b', 'g', 'm'])
    bars = plt.bar(arr3, meanarr, color = colarr[cnt+1] , width = 0.2 )
    plt.errorbar(arr3 , meanarr , yerr = stdarr, fmt = 'o', markersize = 0, barsabove = True, capsize = 5,elinewidth=2, color = 'k')
    plt.legend(handles=colorarr)

    plt.title("{}".format(topofiles[i]))
    plt.ylabel("Fraction of\nparameter sets")
	
    
    count = 0
    #for bar in bars:
    #    height = bar.get_height()
    #    yval = np.round(meanarr[count]*1000)/1000    
    #    plt.text((2*bar.get_x()+  bar.get_width() )/2, yval + .01, "{:.1f}%".format(yval*100), ha='center', va='bottom', fontsize = 15) 
    #    count+=1    
    plt.ylim([0,max(meanarr)+0.1])
    
    cnt+=1
    
  plt.xticks([1,2,3,4], ['Monostable' , 'Bistable' , 'Tristable' , 'Tetrastable'], rotation=45, fontsize=14)      
  plt.tight_layout()
  plt.savefig("stabplot/{}_stabplot.png".format(topofiles[i]), transparent = True)
    
  plt.clf()

