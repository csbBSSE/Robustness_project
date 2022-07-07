from os.path import splitext, isfile, join
from os import listdir
import networkx as nx
import numpy as np
import modules.metric as metric
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import seaborn
from scipy import stats

print('imported modules')

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '25', 'axes.titlesize':'25', 'axes.labelsize':'25', 'xtick.labelsize':'20', 'ytick.labelsize':'20', 'legend.fontsize':'20'})
plt.rcParams['figure.dpi'] = 500
topofiles= [splitext(f)[0] for f in listdir("topofiles_jsd/") if isfile(join("topofiles_jsd/", f))]
topofiles.sort()

plast = []
avgarr = []

pflplast = np.array([0,0])
nflplast =np.array([0,0])

pflplastarr_group = [  [ ] ,  [ ]]
nflplastarr_group = [  [ ] ,  [ ]]

pfljsdarr_group = [  [ ] ,  [ ]]
nfljsdarr_group = [  [ ] ,  [ ]]


pfljsd = np.array([0,0])
nfljsd = np.array([0,0])

for i in range(len(topofiles)):
    network_name = topofiles[i]
    g = metric.networkx_graph(network_name)
    cycle_info = metric.cycle_info(network_name, g)
    lm = metric.matrix(network_name)

    
    pjsd = np.mean(np.loadtxt("raw_data/cnt2_{}_jsd.txt".format(network_name)))

    avgarr.append(pjsd)

jsdmed = np.median(avgarr)
for i in range(len(topofiles)):
    network_name = topofiles[i]
    g = metric.networkx_graph(network_name)
    cycle_info = metric.cycle_info(network_name, g)    

    if(avgarr[i] < jsdmed):
            pfljsd[0]+= cycle_info[0]
            nfljsd[0]+= cycle_info[1]
            pfljsdarr_group[0].append(cycle_info[0])
            nfljsdarr_group[0].append(cycle_info[1])
            
            
    else:
            pfljsd[1]+= cycle_info[0]
            nfljsd[1]+= cycle_info[1]

            pfljsdarr_group[1].append(cycle_info[0])  
            nfljsdarr_group[1].append(cycle_info[1])  
            
n1 = len(plast)/2


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)

pfljsdarr_group= np.array(pfljsdarr_group)
nfljsdarr_group= np.array(nfljsdarr_group)
data_matrix = [pfljsdarr_group[0],pfljsdarr_group[1] , nfljsdarr_group[0] , nfljsdarr_group[1] ]
print(stats.ttest_ind(pfljsdarr_group[0], pfljsdarr_group[1]))
print(stats.ttest_ind(nfljsdarr_group[0], nfljsdarr_group[1]))

labels = ["PFL Left", "PFL Right" , "NFL Left" , "NFL Right"]
r = 2
fig,ax = plt.subplots()
ax = seaborn.violinplot( data = data_matrix ,inner=None , bw = 0.7, cut=0 ,palette=['r','b','r','b'] , linewidth = 4)

hatch = ['*','x','*','x']
ihatch = iter(hatch)
_ = [i.set_hatch(next(ihatch)) for i in ax.get_children() if isinstance(i, mpl.collections.PolyCollection)]
ax.set_xticklabels(labels,fontsize = 25)

a1 = np.mean(data_matrix[0])
print(data_matrix[0])
a2 = np.mean(data_matrix[1])
a3 = np.mean(data_matrix[2])
a4 = np.mean(data_matrix[3])
plt.scatter([0,1,2,3], [a1,a2,a3,a4] , c='k' )
#plt.title("Feedback Loops (Network  Size 4) (Grouped by JSD)" , c= '0.3' , fontweight = 'bold', fontsize = 25)
plt.xlabel("(Grouped by JSD)")

ax.set_ylabel("Avg. Perturbation JSD" ,fontsize = 32)
f=2*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)

plt.savefig("violinplot_jsd.png", transparent = True)




