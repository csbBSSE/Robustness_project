import os
import time
from os import listdir
from os.path import isfile, join
topofiles= [os.path.splitext(f)[0] for f in listdir("output/") if isfile(join("output/", f))]

import numpy as np

q=0

for i in topofiles:
    q+=1
    t=len(i)
    if(i[t-1]!='l'):
        os.remove("output/{}.txt".format(i))
    if(q%2000==0):
        print(q,i)