import numpy as np
import matplotlib.pyplot as plt
from math import pow
import os
from scipy.stats import norm, zscore
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
#from scipy.stats import kde
# from scipy import stats

from matplotlib import rcParams

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





network_name =  params['input_filenames'][0] # name_solution.dat and name_async_unweigh_ssprob_all.txt files
plot_plotterdata = 0 # if boolean plot values are included or not

#NOTE ANISH MODIFICATION: if you have 4 genes, and you want the last three, set left to 1 and right to 3 yeah?


left= params['constant_node_count'][0]   
right=len(id_to_node)-1

### CHANGE WAS MADE: it reads racipe files from RACIPE/ and boolean files from output/, its more automated this way.




#give input as new solution file created by racipe

data=np.loadtxt("{}_solution.dat".format(network_name))[:,2:]

#give column range of genes you want to classify, 0 indexed
# print(data)
# exit(0)
length=right-left+1

# datacol=[data[:,0]]*(length)
# for u in range(left,right+1):
#     datacol[u-left]=data[:,u]
datacol = [data[:,u] for u in range(left,right + 1)]
print("dataloaded")
# zscoredx=[datacol[0]]*(length)
# for u in range(0,length):
#     zscoredx[u]=stats.zscore(datacol[u])

zscoredx = [zscore(datacol[u]) for u in range(0,length)]


mean=[0]*length

for i in range(0,right+1):
    mean[i]=np.mean(data[i])
print("zscore done")
#
# kdefitx=[sm.nonparametric.KDEUnivariate(zscoredx[0])]*(length)
# for u in range(0,length):
#     kdefitx[u]=sm.nonparametric.KDEUnivariate(zscoredx[u])
#     kdefitx[u].fit(bw=0.1)
kdefitx = [sm.nonparametric.KDEUnivariate(zscoredx[u]) for u in range(0,length)]
for u in range(0,length):
    kdefitx[u].fit(bw = 0.1)
print("kdefit done")

pivot=[0]*length
pivotpos=[0]*length
n=len(kdefitx[0].support)


# racipeclassify=np.zeros(2**length)
racipeclassify = {}

for i in range(len(zscoredx[0])):
    zarr=[0]*length
    power=int(2**(length-1) +1e-9)
    index=int(0)
    for u in range(length):
        
        #print(data[u+left][i],u+left,i)
        zarr[u]=int(zscoredx[u][i] >0)
        index+=power*zarr[u]
        power=power/2
    # racipeclassify[int(index)]+=1
    index = int(index)
    try:
        racipeclassify[index] += 1
    except:
        racipeclassify[index] = 1

# yaxis=[0]*(2**length)
# xaxislabel=[str("")]*(2**length)

dividend = len(zscoredx[0])
for u in racipeclassify.keys():
    racipeclassify[u] = racipeclassify[u] / dividend

# for u in range(2**length):
#     q=u
#     k=0
#     while(k<length):
#         xaxislabel[u]+=str(int(q%2) )
#         q=q/2
#         k+=1
#     xaxislabel[u] = "".join(reversed(xaxislabel[u]))
#     yaxis[u]=racipeclassify[u]/len(zscoredx[0])

binlabelformat = "{0:0" + str(length) +  "b}"




final_index = []
# for i in range(len(yaxis)):
#     if yaxis[i] >= 0.01:
#         final_index.append(i)
racipe_probfilefull = open("Datafiles/{}_racipe_probfull.txt".format(network_name), 'w')
for i in racipeclassify.keys():
    if racipeclassify[i] >= 0.01:
        final_index.append(i)
    label = binlabelformat.format(i)    
    racipe_probfilefull.write("{} {:.6f}\n".format(label, racipeclassify[i]))  

racipe_probfile = open("Datafiles/{}_racipe_prob.txt".format(network_name), 'w')
for i in final_index:
    label = binlabelformat.format(i)
    racipe_probfile.write("{} {:.6f}\n".format(label, racipeclassify[i]))
    

# print(np.array([yaxis[i] for i in final_index]), np.array([yaxis[i] for i in final_index]).argsort())
final_yaxis = [racipeclassify[i] for i in final_index]
final_xaxislabel = [binlabelformat.format(i) for i in final_index]
argarr = np.argsort(np.array(final_yaxis))[::-1]
# print(argarr)
final_yaxis = [final_yaxis[i] for i in argarr]
final_xaxislabel = [final_xaxislabel[i] for i in argarr]
# final_xaxis=np.array([i for i in range(1,len(final_yaxis)+1)])
final_xaxis = np.linspace(0, len(final_yaxis), len(final_yaxis))
plt.figure(figsize=(20,12))
# print(len(final_yaxis), len(final_xaxislabel), len(final_xaxis))
if(not(plot_plotterdata)):
    rcParams.update({'figure.autolayout':True})
    plt.bar(final_xaxis,final_yaxis, width = 0.3, color = 'r')
    plt.xticks(final_xaxis, final_xaxislabel, rotation = 'vertical')
    # plt.show()
    plt.savefig("Graphs/{}_racipe.png".format(network_name))

# print(argarr)
# print(yaxis)




if plot_plotterdata:
    dataplotter = open("output/{}_ssprob_all.txt".format("{}_async_unweigh".format(network_name)),'r').read().split("\n")[1:-1]
    x_labels = []
    y_data = []
    # print(dataplotter)
    for temp in dataplotter:
        # print(i)
        i = temp.split(" ")
        x_labels.append(i[0])
        y_data.append(float(i[1]))
    final_yaxislist=list(final_yaxis)
    list_labels=[]
    dict_yaxis={}
    for i in range(len(final_xaxis)):
        list_labels.append(final_xaxislabel[i])
        dict_yaxis.update( {final_xaxislabel[i] : i })
    cnt=0
    for i in range(len(y_data)):
        if (y_data[i] >= 0.01 and (str(x_labels[i]) not in final_xaxislabel) ):
            list_labels.append(x_labels[i])
            dict_yaxis.update( {x_labels[i] : len(final_xaxis)+cnt} )
            try:
                final_yaxislist.append(racipeclassify[ int(x_labels[i],2) ])
            except:
                final_yaxislist.append(0)
            cnt+=1

    final_ydata = np.zeros(len(list_labels))
    for i in range(len(y_data)):
        if(x_labels[i] in dict_yaxis):
            index=dict_yaxis[x_labels[i]]
            final_ydata[index]=y_data[i]

    weight = 0.35
    final_xaxis=[i for i in range(len(list_labels)) ]
    rcParams.update({'figure.autolayout':True})
    plt.bar(final_xaxis,np.array(final_yaxislist) , width = 0.3, color = 'r')
    plt.bar(np.array(final_xaxis)+weight, final_ydata, width = 0.3, color = 'b')
    plt.xticks(final_xaxis,list_labels, rotation = 'vertical')
    plt.title( "JSD = {}".format(jensenshannon(final_ydata,np.array(final_yaxislist),2 )))
    # plt.show()
    plt.savefig("{}_racipeVbool.png".format(network_name))
    # print(final_ydata)
# print(xaxislabel)





