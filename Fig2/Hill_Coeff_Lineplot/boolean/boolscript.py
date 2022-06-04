import os
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import time

def matrix(network_name):
    network = open("input/{}.topo".format(network_name)).read().split("\n")[1:]
    ids = open("input/{}.ids".format(network_name)).read().split("\n")[1:]

    empty_break = 1
    while empty_break:
        if "" in network:
            network.remove("")
        else:
            empty_break = 0

    empty_break = 1
    while empty_break:
        if "" in ids:
            ids.remove("")
        else:
            empty_break = 0

    nodes_dict = {}
    for i in ids:
        temp = i.split(" ")
        nodes_dict[temp[0]] = int(temp[1])

    nodes = len(ids)
    link_matrix = np.zeros((nodes, nodes))

    for i in network:
        temp = i.split(" ")
        source = nodes_dict[temp[0]]
        target = nodes_dict[temp[1]]

        link_matrix[source][target] = -1 if temp[2] == '2' else 1

    return link_matrix
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()

nodes = [0]*len(topofiles)
for i in range(len(topofiles)):
    nodes[i] = matrix(topofiles[i]).shape[0]

inittext = """input_folder_name input
output_folder_name output
input_filenames {}
num_runs 3
num_simulations {}
maxtime 2000
constant_node_count 0
"""

num_simulations = 5000


for i in range(len(topofiles)):
    curnetwork = open("curnetwork.txt", 'w')
    looptime = time.time()
    initfile = open("init.txt", "w")
    initfile.write(inittext.format(topofiles[i], num_simulations * nodes[i]))
    initfile.close()

    print("Bool_{}".format(topofiles[i]) )
    os.system("./mainbool")

    os.system("python3 plotter.py")

    curnetwork.write(topofiles[i])
    curnetwork.close()


    for runs in range(1, 4):
        os.remove("output/{}_ss_run{}.txt".format(topofiles[i], runs))
        os.remove("output/{}_nss_run{}.txt".format(topofiles[i], runs))
        os.remove("output/{}_init_run{}.txt".format(topofiles[i], runs))
    cnt = 1
