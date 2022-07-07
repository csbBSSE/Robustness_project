import numpy as np
import matplotlib.pyplot as plt
from math import pow
import os
from scipy.stats import norm, zscore
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from matplotlib import rcParams
import matplotlib.lines as mlines
#from scipy.stats import kde
# from scipy import stats
import matplotlib
from os import listdir
from os.path import isfile, join
import matplotlib.patches as mpatches
from matplotlib import rcParams
#matplotlib.use('ps')
#from matplotlib import rc

#rc('text',usetex=True)
#rc('text.latex', preamble='\usepackage{color}')
topofiles= os.listdir()
matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '25', 'axes.titlesize':'20', 'axes.labelsize':'20', 'xtick.labelsize':'20', 'ytick.labelsize':'15', 'legend.fontsize':'15'})

#rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', })
#print(topofiles)
version='cont'


if "maincurrent.c" in topofiles:
    version='ising'
else:
    version = 'cont'

####
import initialise.initialise as initialise
import initialise.parser as parser
in_file = 'init.txt'
max_initlines = 14
begin=1
process_count=1
params = initialise.initialise(in_file, max_initlines)
params['file_reqs'] = initialise.set_file_reqs(params)
id_to_node=[]

for i in params['file_reqs']:
        for j in params['input_filenames']:
                random_seed = int(begin) + process_count
                weighted_tick = 1 if "_weigh" in i else 0
                async_tick = 1 if "_async" in i else 0
                link_matrix, id_to_node = parser.parse_topo(params,j,weighted_tick, random_seed)
 #####





orig_network_name =  params['input_filenames'][0] # name_solution.dat and name_async_unweigh_ssprob_all.txt files
plot_plotterdata = 0 # if boolean plot values are included or not

left= params['constant_node_count'][0]
right=len(id_to_node)-1

#give column range of genes you want to classify, 0 indexed
# print(data)
# exit(0)
length=right-left+1



binlabelformat = "{0:0" + str(length) +  "b}"


runs = [100,1000,10000]
cnt = -1
final_xaxis=[]

allowed_list = []


colorarr = []
c = ['r' , 'b', 'g', 'm']
labels = ['100 ODEs' , '1000 ODEs' , '10000 ODEs']

for i in range(3):
    colorarr.append( mpatches.Patch(color=c[i], label=labels[i]) )



for j in runs:

    network_name = "{}_{}".format(params['input_filenames'][0],j)
    
    rac_probfile=open("Datafiles_error/{}_racipe_probfull_error.txt".format(network_name))

    rac_probdata = rac_probfile.read().split("\n")[0:]

    if "" in rac_probdata:
        rac_probdata.remove("")
    rac_probfile.close()

    xaxislabel= []
    final_xaxislabel=[]
    yaxis = []
    final_yaxis =[]
    error = []
    final_error = []
    xaxis=[]
    final_xaxis=[]
    final_index=[]
    racipeclassify = {}
    dict_encounter={}

    encount=0
    probval=0
    for k in rac_probdata:
        temp = k.split(" ")

        if(temp[0][left:] not in xaxislabel):
                xaxislabel.append(temp[0][left:])
                encount+=1
                dict_encounter.update( {temp[0][left:] : encount-1 })
                yaxis.append(float(temp[1]))
                error.append(float(temp[2]))

                probval=float(temp[1])
        else:
                yaxis[  dict_encounter[ temp[0][left:] ] ] += float(temp[1])
                error[dict_encounter[temp[0][left:]]] += float(temp[2])
                probval = yaxis[  dict_encounter[ temp[0][left:] ] ]


    for k in range ( len(yaxis) ):

        if((yaxis[k] >=0.02 and cnt == -1)  or (cnt > -1 and xaxislabel[k] in allowed_list) ):
            final_xaxislabel.append(xaxislabel[k])
            allowed_list.append(xaxislabel[k])
        
            final_index.append(int(xaxislabel[k],2))

        racipeclassify[ int(xaxislabel[k],2) ]  = [float(yaxis[k]), float(error[k])]


    final_yaxis = [racipeclassify[i][0] for i in final_index]
    final_xaxislabel = [binlabelformat.format(i) for i in final_index]
    final_error = [racipeclassify[i][1] for i in final_index]

    argarr = np.argsort(np.array(final_yaxis))[::-1]

    final_yaxis = [final_yaxis[i] for i in argarr]
    final_error = [final_error[i] for i in argarr]
    final_xaxislabel = [final_xaxislabel[i] for i in argarr]
    final_xaxis = np.linspace(0, len(final_yaxis), len(final_yaxis))


    #print(final_yaxis)
    #print(final_xaxislabel)

    if(not(plot_plotterdata)):
        rcParams.update({'figure.autolayout':True})
        plt.bar(final_xaxis + cnt*0.25,final_yaxis,  width = 0.2, color = c[cnt+1] )
        plt.errorbar(final_xaxis + cnt*0.25 ,final_yaxis, yerr = final_error , color = 'k', fmt = 'o', markersize = 0, barsabove = True, capsize = 3)

        # plt.show()

    cnt+=1
    
plt.title("{}".format(orig_network_name))
plt.xticks(final_xaxis, final_xaxislabel, rotation = 'vertical', fontsize = 14)
plt.legend(handles=colorarr)      
plt.tight_layout()
plt.savefig("distrplot/{}_distrplot.png".format(orig_network_name), transparent = True)
plt.clf()
