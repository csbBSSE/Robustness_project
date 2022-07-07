import numpy as np
import matplotlib.pyplot as plt
from math import pow
import os
from copy import copy
from scipy.stats import norm, zscore

import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from statsmodels.stats.weightstats import DescrStatsW
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

numsplit = 3

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
discard_cols = 3



#NOTE ANISH MODIFICATION: if you have 4 genes, and you want the last three, set left to 1 and right to 3 yeah?


left= params['constant_node_count'][0]
right=len(id_to_node)-1


### CHANGE WAS MADE: it reads racipe files from RACIPE/ and boolean files from output/, its more automated this way.




#give input as new solution file created by racipe

data=np.loadtxt("{}_solution.dat".format(network_name))[:,discard_cols - 1:]
weights_states = data[:,0]
data = data[:, 1:]

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


zscoredx = [0]* length
for u in range(length):
    weighted_mean = DescrStatsW(datacol[u], weights = weights_states, ddof = 0)
    zscoredx[u] = (datacol[u] - weighted_mean.mean)/weighted_mean.std

# mean=[0]*length
#
# for i in range(0,right+1):
#     mean[i]=np.mean(data[i])
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
n = len(kdefitx[0].support)


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
        racipeclassify[index] += weights_states[i]
    except:
        racipeclassify[index] = weights_states[i]

# yaxis=[0]*(2**length)
# xaxislabel=[str("")]*(2**length)

dividend = sum(weights_states)
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

for i in racipeclassify.keys():
    if racipeclassify[i] >= 0.01:
        final_index.append(i)



datasplit = []
weightsplit = []
for k in range(numsplit - 1):
    a = []
    for j in datacol:
        size = j.shape[0]//numsplit
        a.append(j[k * size : (k+1) * size])
    datasplit.append(a)
a = []
for j in datacol:
    size = j.shape[0]//numsplit
    a.append(j[(numsplit - 1) * size:])
datasplit.append(a)

size = len(weights_states)//numsplit
for i in range(numsplit - 1):
    weightsplit.append(weights_states[i * size : (i + 1) * size])
weightsplit.append(weights_states[(numsplit - 1) * size:])

#for i in range(numsplit):
#    print(len(datasplit[i][0]), len(weightsplit[i]))

racipeclassify_all = [0] * (numsplit + 1)
for i in range(numsplit + 1):
    racipeclassify_all[i] = copy(racipeclassify)

for splititer in range(numsplit):
    zscoredx = [0]* length
    for u in range(length):
        weighted_mean = DescrStatsW(datasplit[splititer][u], weights = weightsplit[splititer], ddof = 0)
        zscoredx[u] = (datasplit[splititer][u] - weighted_mean.mean)/weighted_mean.std

    # mean=[0]*length
    #
    # for i in range(0,right+1):
    #     mean[i]=np.mean(data[i])
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
    n = len(kdefitx[0].support)


    # racipeclassify=np.zeros(2**length)
    # racipeclassify = {}
    for i in racipeclassify_all[splititer + 1].keys():
        racipeclassify_all[splititer + 1][i] = 0
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
            racipeclassify_all[splititer + 1][index] += weightsplit[splititer][i]
        except:
            racipeclassify_all[splititer + 1][index] = weightsplit[splititer][i]

    # yaxis=[0]*(2**length)
    # xaxislabel=[str("")]*(2**length)

    dividend = sum(weightsplit[splititer])
    for u in racipeclassify_all[splititer + 1].keys():
        racipeclassify_all[splititer + 1][u] = racipeclassify_all[splititer + 1][u] / dividend

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

error_final = {}
for i in racipeclassify_all[0].keys():
    nums = [racipeclassify_all[j][i] for j in range(1, numsplit + 1)]
    error_final[i] = np.std(nums)


racipe_probfilefull = open("Datafiles_error/{}_racipe_probfull_error.txt".format(network_name), 'w')
for i in racipeclassify.keys():
    label = binlabelformat.format(i)
    racipe_probfilefull.write("{} {:.6f} {:.6f}\n".format(label, racipeclassify[i], error_final[i]))


racipe_probfile = open("Datafiles_error/{}_racipe_prob_error.txt".format(network_name), 'w')
for i in final_index:
    label = binlabelformat.format(i)
    racipe_probfile.write("{} {:.6f} {:.6f}\n".format(label, racipeclassify[i], error_final[i]))

racipe_probfile.close()
racipe_probfilefull.close()

# combined_final_index = set({})
# for i in range(numsplit):
#     for j in final_index_all[i]:
#         combined_final_index.append(j)
#
# combined_final_index = list(combined_final_index)
# del final_index_all
#
# for i in range(numsplit):
#     for j in combined_final_index:
#         if racipeclassify_all[i][j]:
#             continue
#         else:
#             racipeclassify_all[i][j] = 0
#
#
# racipeclassify_final = {}
# error_final = {}
# for i in combined_final_index:
#     nums = [racipeclassify_all[j][i] for j in range(numsplit)]
#     racipeclassify_final[i] = np.mean(nums)
#     error_final[i] = np.std(nums)
#
# combined_total_index = set({})
# for i in range(numsplit):
#     for j in racipeclassify_all[i].keys():
#         combined_total_index.append(j)
#
# for i in range(numsplit):
#     for j in combined_total_index:
#         if racipeclassify_all[i][j]:
#             continue
#         else:
#             racipeclassify_all[i][j] = 0

