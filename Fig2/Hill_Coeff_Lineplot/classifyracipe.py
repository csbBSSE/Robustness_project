import numpy as np
import matplotlib.pyplot as plt
from math import pow
import os
from scipy.stats import norm, zscore
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from statsmodels.stats.weightstats import DescrStatsW
from matplotlib import rcParams
import initialise.initialise as initialise
import initialise.parser as parser
import sys

final_file_name = sys.argv
in_file = 'init.txt'
begin=1
process_count=1
params = initialise.initialise(in_file)
initialise.create_folders(params)
id_to_node=[]


for j in params['input_filenames']:
    random_seed = int(begin) + process_count
    link_matrix, id_to_node = parser.parse_topo(params,j,weighted_tick, random_seed)


network_name =  params['input_filenames'][0] # name_solution.dat and name_ssprob_all.txt files
plot_plotterdata = 0 # if boolean plot values are included or not
discard_cols = 3

left= params['constant_node_count'][0]
right=len(id_to_node)-1

#give input as new solution file created by racipe

data=np.loadtxt("{}_solution.dat".format(network_name))[:,discard_cols - 1:]
weights_states = data[:,0]
data = data[:, 1:]

#give column range of genes you want to classify, 0 indexed
length=right-left+1

datacol = [data[:,u] for u in range(left,right + 1)]
print("dataloaded")

zscoredx = [0]* length
for u in range(length):
    weighted_mean = DescrStatsW(datacol[u], weights = weights_states, ddof = 0)
    zscoredx[u] = (datacol[u] - weighted_mean.mean)/weighted_mean.std

print("zscore done")

pivot=[0]*length
pivotpos=[0]*length

racipeclassify = {}

for i in range(len(zscoredx[0])):
    zarr=[0]*length
    power=int(2**(length-1) +1e-9)
    index=int(0)
    for u in range(length):

        zarr[u]=int(zscoredx[u][i] >0)
        index+=power*zarr[u]
        power=power/2
    index = int(index)
    try:
        racipeclassify[index] += weights_states[i]
    except:
        racipeclassify[index] = weights_states[i]

dividend = sum(weights_states)
for u in racipeclassify.keys():
    racipeclassify[u] = racipeclassify[u] / dividend

binlabelformat = "{0:0" + str(length) +  "b}"




final_index = []
racipe_probfilefull = open("Datafiles/{}_racipe_probfull_{}_{}.txt".format(network_name, final_file_name[1], final_file_name[2]), 'w')

for i in range(2**length):
    label = binlabelformat.format(i)
    try:
        racipeclassify[i]
    except:
        racipeclassify[i] = 0

    racipe_probfilefull.write("{} {:.6f}\n".format(label, racipeclassify[i]))