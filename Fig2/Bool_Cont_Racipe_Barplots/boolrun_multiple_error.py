import os
import time
from os import listdir
from os.path import isfile, join
import sys

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]

import numpy as np
print(topofiles)

####
import initialise.initialise as initialise
import initialise.parser as parser
in_file = 'init.txt'
begin=1
process_count=1
params = initialise.initialise(in_file)
id_to_node=[]
link_matrix = [0]*len(topofiles)
copy_linkmatrix = [0]*len(topofiles)
id_to_node = [0]*len(topofiles)
length = len(topofiles)
nodes = [0]*len(topofiles)

for j in range (len(topofiles)):
                repj=topofiles[j]
                random_seed = int(begin) + process_count
                print(topofiles[j])
                link_matrix[j], id_to_node[j] = parser.parse_topo(params,repj,random_seed)
                copy_linkmatrix[j], id_to_node[j] = parser.parse_topo(params,repj,random_seed)
                nodes[j] = len(id_to_node[j])
#####


num_simulations = 10000

inittext = """input_folder_name input
output_folder_name output
input_filenames {}
num_runs 3
num_simulations {}
maxtime 2000
constant_node_count 0
"""

tottime = time.time()
version = 'bool' ###change this to cont or bool as needed

for i in range(len(topofiles)):
    curnetwork = open("curnetwork.txt", 'w')
    looptime = time.time()
    initfile = open("init.txt", "w")
    initfile.write(inittext.format(topofiles[i], num_simulations * nodes[i]))
    initfile.close()

    print("{}_{}".format(version,topofiles[i]) )
    os.system("./main{}".format(version))   
    os.system("python3 plotter.py")
    
    
    curnetwork.write(topofiles[i])
    curnetwork.close()
    
    os.system("python3 racvsbool2_error.py")
    
    for j in range(1,4):
        os.remove("output/{}_init_run{}.txt".format(topofiles[i],j))
        os.remove("output/{}_nss_run{}.txt".format(topofiles[i],j))
        os.remove("output/{}_ss_run{}.txt".format(topofiles[i],j))