import os
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()

num_sim = 5000
num_split = 1

# REMEMBER TO MODIFY THREADS
racipecmd = "./RACIPE {}.topo {}.ids -minN {} -maxN {} -threads 50 -num_paras {} -num_stability 4"

def run_i(i, min, max, topofile, run_no):
    global num_sim
    print("x{}_{}".format(i, run_no))
    os.system(racipecmd.format(topofile, topofile, min, max, num_sim))
    os.system('python3 classifyracipe.py {} {}'.format(i, run_no))

minall = [5 * i + 1 for i in range(0, 4)]
maxall = [5 * i + 6 for i in range(0, 4)]

for i in topofiles:
    for j in range(4):
        for k in range(3):
            run_i(j + 1, minall[j], maxall[j], i, k + 1)
