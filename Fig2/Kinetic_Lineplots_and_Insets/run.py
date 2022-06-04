import os
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()

amplifications = [1, 2, 3, 0.33, 0.5]
amplifications.sort()
dirs = ["_all", "_nohill", "_onlyhill"]
jsddir = ["JSD" + i for i in dirs]
plastdir = ["PLAST" + i for i in dirs]
write_str = "{:.6f} " * 5 + "{:.6f}\n"
try:
	os.mkdir("ALL")
except:
	print("dir ALL exists")
	
for i in topofiles:
	fin_jsd = open("ALL/{}_JSD.txt".format(i), 'w')
	fin_plast = open("ALL/{}_PLAST.txt".format(i), 'w')
	
	for amp in amplifications:
		jsdvals = []
		plastvals = []
		for dircnt in range(len(dirs)):
			jsdfile = np.loadtxt("{}/{}_{}_jsd.txt".format(jsddir[dircnt], i, amp))
			plastfile = np.loadtxt("{}/{}_{}_plast.txt".format(plastdir[dircnt], i, amp))
			jsdvals.append(np.mean(jsdfile))
			jsdvals.append(np.std(jsdfile))
			plastvals.append(np.mean(plastfile))
			plastvals.append(np.std(plastfile))
		fin_jsd.write(write_str.format(*jsdvals))
		fin_plast.write(write_str.format(*plastvals))
	fin_jsd.close()
	fin_plast.close()
