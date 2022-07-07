import os
import time
from os import listdir
from os.path import isfile, join
import shutil
import sys

topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]

import numpy as np
print(topofiles)

def racipe_run(topofile, run):
	os.system("./RACIPE {}.topo {}.ids -num_ode {} -num_paras 10000 -num_stability 4 -threads 60".format(topofile, topofile, run))

for i in range(len(topofiles)):
    runs = [10,100,1000,10000]
    for j in runs:
    	shutil.copyfile("input/{}.topo".format(topofiles[i]), "{}_{}.topo".format(topofiles[i], j))
    	shutil.copyfile("input/{}.topo".format(topofiles[i]), "input/{}_{}.topo".format(topofiles[i], j))
    	shutil.copyfile("input/{}.ids".format(topofiles[i]), "{}_{}.ids".format(topofiles[i], j))
    	shutil.copyfile("input/{}.ids".format(topofiles[i]), "input/{}_{}.ids".format(topofiles[i], j))
    	racipe_run("{}_{}".format(topofiles[i], j), j)
		
