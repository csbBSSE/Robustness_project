import os
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': 20})



for nn in topofiles:
    if(nn == 'GRHL2'):
        continue
    jsd = np.loadtxt("ALL/{}_JSD.txt".format(nn))
    plast = np.loadtxt("ALL/{}_PLAST.txt".format(nn))

    amplifications = [1, 2, 3, 0.33, 0.5]
    amplifications.sort()

    colours = ["r", "g", "b"]
    legendarr = ["All Parameters", "No Hill", "Only Hill"]

    for i in range(len(colours)):
        plt.figure(2)
        plt.plot(amplifications, jsd[:, 2 * i], c = colours[i] ,linewidth = 3)# , marker = markers[markercount]
        plt.errorbar(amplifications, jsd[:, 2 * i], yerr = jsd[:, 2 * i + 1], c = colours[i], fmt = 'o', markersize = 0,  linewidth = 2,barsabove = True, capsize = 5, ecolor = 'k')

        plt.figure(1)
        plt.plot(amplifications, plast[:, 2 * i], c = colours[i],linewidth = 3)# , marker = markers[markercount]
        plt.errorbar(amplifications, plast[:, 2 * i], yerr = plast[:, 2 * i + 1], c = colours[i], fmt = 'o', markersize = 0, linewidth = 2, barsabove = True, capsize = 5 , ecolor = 'k')

    fig = plt.figure(1)

    r = 2
    matplotlib.rcParams.update({'font.size': 12*r})
    plt.rc('legend',fontsize=12*r)

    plt.xticks(fontsize= 12*r)
    plt.yticks(fontsize= 12*r)

    plt.xlabel("Multiplication factor for maximum range\n({})".format(nn),fontsize=15*r, fontweight = "bold", c='0.3')
    plt.ylabel("Plasticity",fontsize=15*r, fontweight = "bold", c='0.3')
    #plt.title("{}".format(nn), fontweight = "bold", c='0.3')

    legend = plt.legend(legendarr, frameon = False)
    plt.setp(legend.get_texts(), color='0.3',fontsize = 12*r , fontweight="bold" )
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)
    plt.tight_layout()
    
    
    
    
    plt.savefig("{}_paramvar_plast.png".format(nn) , transparent = True)
    plt.close()


    fig = plt.figure(2)
    ax = fig.axes[0]

    r = 2
    matplotlib.rcParams.update({'font.size': 12*r})

    plt.xticks(fontsize = 12*r)
    plt.yticks(fontsize = 12*r)

    
    plt.xlabel("Multiplication factor for maximum range\n({})".format(nn), fontsize=15*r,  fontweight = "bold", c='0.3')
    plt.ylabel("JSD from WT", fontsize=15*r, fontweight = "bold", c='0.3')
    #plt.title("{}\n\n\n".format(nn), fontweight = "bold", c='0.3')
 
    
    if(nn!= 'NRF2'):
        legend = plt.legend(legendarr, frameon = False)
        
    else:
         legend = plt.legend(legendarr,bbox_to_anchor=(0.4, 0.7), frameon = False)
    plt.setp(legend.get_texts(), color='0.3', fontweight="bold" )    
    f=r*np.array(plt.rcParams["figure.figsize"])
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(f)
    plt.tight_layout()    
    plt.savefig("{}_paramvar_jsd.png".format(nn) , transparent = True)
    plt.close()
