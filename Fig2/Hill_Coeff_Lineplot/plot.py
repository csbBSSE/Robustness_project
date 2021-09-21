import os
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import jensenshannon
import matplotlib

matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size':'18'})
plt.figure(figsize=(11,6))

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()


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


jsdmean = []
jsdavg = []
width = 0.4
sets = 10
x = np.array([i for i in range(sets)]) * 3
for i in range(len(topofiles)):
    jsdmatrix = np.zeros((sets,3)) # each row is a param range set, cols are runs
    for run in range(3):
        booldatafile = open("boolean/output/{}_ssprob_run{}.txt".format(topofiles[i], run + 1)).readlines()
        try:
            booldatafile.remove("")
        except:
            pass
        n = matrix(topofiles[i]).shape[0]
        boolclassify = {}
        stringbool = '{0:0' + str(n) + 'b}'
        for iter in range(2 ** n):
            boolclassify[stringbool.format(iter)] = 0
        booldata = []
        for temp in booldatafile:
            temp2 = temp.split(" ")
            boolclassify[temp2[0]] = float(temp2[1])
        for iter in range(2 ** n):
            booldata.append(boolclassify[stringbool.format(iter)])
        print(booldata)
        
        for parrange in range(sets):
            racdata = np.loadtxt("Datafiles/{}_racipe_probfull_{}_{}.txt".format(topofiles[i], parrange + 1,  run + 1))
            racdata = racdata[racdata[:, 0].argsort()][:, 1]
            jsdmatrix[parrange][run] = jensenshannon(booldata, racdata)
    mean = [np.mean(jsdmatrix[p, :]) for p in range(sets)]
    std = [np.std(jsdmatrix[p, :]) for p in range(sets)]

    plt.scatter(x, mean, label = topofiles[i]) # add ", width = width / 2" for bar plots
    plt.errorbar(x, mean, yerr = std) # , fmt = 'none'
    # ADD " + width * i" for bar plots 
xtickarr = ["{}-{}".format(2 * i + 1, 2 * i + 2) for i in range(0,sets)]
for i in range(len(xtickarr)):
    if i%2 == 1:
        xtickarr[i] = ''
plt.legend()

leg = plt.legend(bbox_to_anchor=(0.8,0.78), framealpha = 1.0)
leg.get_frame().set_edgecolor('k')
leg.get_frame().set_linewidth(3)

plt.xlabel("Hill Coefficient Ranges", fontsize = 18)
plt.ylabel("JSD from Ising", fontsize = 18)
plt.ylim([0.18, 0.55])
plt.xticks(x, xtickarr)
plt.title("JSD on Varying Hill Coefficient  ")
plt.savefig('jsdlineplot.png', transparent = True)