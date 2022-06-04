import numpy as np
import matplotlib.pyplot as plt
from math import pow
import os
from scipy.stats import norm, zscore
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
import sys
#from scipy.stats import kde
# from scipy import stats

from matplotlib import rcParams

####
import initialise.initialise as initialise
import initialise.parser as parser
in_file = sys.argv[1]
max_initlines = 14
begin=1
process_count=1
params = initialise.initialise(in_file, max_initlines)
params['file_reqs'] = initialise.set_file_reqs(params)
id_to_node=[]
probfile_dir = sys.argv[2]

for i in params['file_reqs']:
        for j in params['input_filenames']:
                random_seed = int(begin) + process_count
                weighted_tick = 1 if "_weigh" in i else 0
                async_tick = 1 if "_async" in i else 0
                link_matrix, id_to_node = parser.parse_topo(params,j,weighted_tick, random_seed)
 #####





network_name =  params['input_filenames'][0] # name_solution.dat and name_async_unweigh_ssprob_all.txt files


length=len(id_to_node)

binlabelformat = "{0:0" + str(length) +  "b}"




jsdfile = open("JSD/{}_jsd.txt".format(probfile_dir), 'a')

bool_probfilefull = open("Datafiles/{}_ising_probfull.txt".format(network_name), 'w')

probfile=open("output/{}_async_unweigh_ssprob_all.txt".format(network_name))
probdata = probfile.read().split("\n")[1:]


boolclassify={}
binlabelformat = "{0:0" + str(length) +  "b}"

if "" in probdata:
   probdata.remove("")
probfile.close()

for k in probdata:
       temp = k.split(" ")
       index=int(temp[0],2)
       boolclassify[index]=float(temp[1])


jsdoutstr = "{} ".format(network_name)

for i in range(2**length):
    label = binlabelformat.format(i)
    try:
        boolclassify[i]
    except:
        boolclassify[i] = 0

    bool_probfilefull.write("{} {:.6f}\n".format(label, boolclassify[i]))
    if i == 2**length - 1:
        jsdoutstr += "{:.6f}".format(boolclassify[i]) + "\n"
    else:
        jsdoutstr += "{:.6f}".format(boolclassify[i]) + " "

jsdfile.write(jsdoutstr)
