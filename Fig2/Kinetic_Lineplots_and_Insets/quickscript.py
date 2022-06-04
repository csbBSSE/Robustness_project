import os
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]
topofiles.sort()

init = """input_folder_name input
output_folder_name output
input_filenames {}
num_runs 1
num_simulations 10000
maxtime 2000
constant_node_count 0
"""
num_sim = 100  ##modify this as needed
# arg2 -minF {} and arg6 -maxF {}

# REMEMBER TO MODIFY THREADS
racipecmd = "./RACIPE {}.topo {}.ids -minN {} -minP {} -minK {} -minF {} -maxN {} -maxP {} -maxK {} -maxF {} -threads 4 -num_paras {} -num_stability 4"
valuemin = np.array([1, 1, 0.1, 1]) # second val 1 foldchange
valuemax = np.array([6, 100, 1, 100]) # second val 100 foldchange

def run_i(i, valuemin, tempvalue, valuemax, topofile, run_no):
    global num_sim
    plast = open('PLAST/{}_{}_plast.txt'.format(topofile, i), 'a')
    print("x{}_{}".format(i, run_no))
    tempvalue = i * np.array(valuemax)
    tempvalue[0] /= i
    os.system(racipecmd.format(topofile, topofile, *valuemin, *tempvalue, num_sim))
    os.system('python3 classifyracipe.py {} {}'.format(i, run_no))
    os.system('python3 racvsbool.py {} {}'.format(i, run_no))
    a = 0
    for tempvar in range(2,5):
        try:
            a += np.matrix(np.loadtxt("{}_solution_{}.dat".format(topofile, tempvar))).shape[0]
            os.remove("{}_solution_{}.dat".format(topofile, tempvar))
        except:
            pass
    a /= num_sim
    plast.write("{:.6f}\n".format(a))
    plast.close()

amplifications = [1, 2, 3, 0.33, 0.5]
amplifications.sort()
markers = [',', '+', '.', 'o', '*']
markercount = 0
colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
colourcount = 1
for i in topofiles:

    for amp in amplifications:
        f1 = open("JSD/{}_{}_jsd.txt".format(i, amp), 'w')
        f1.close()
        f2 = open("PLAST/{}_{}_plast.txt".format(i, amp), 'w')
        f2.close()

    print(i)
    initf = open('init.txt','w')
    initf.write(init.format(i))
    initf.close()

    print("wt")
    tempvalue = np.array(valuemax)
    tempvalue[0] /= 1
    os.system(racipecmd.format(i, i, *valuemin, *tempvalue, num_sim))
    os.system('python3 classifyracipe.py {} {}'.format('wt', '0'))


    for delet in range(2,5):
        try:
            os.remove("{}_solution_{}.dat".format(topofile, delet))
        except:
            pass


    for amp in amplifications:
        for run in range(1, 4):
            run_i(amp, valuemin, tempvalue, valuemax, i, run)

    jsd_mean = []
    jsd_error = []
    plast_mean = []
    plast_error = []

    for amp in amplifications:
        jsdarr = np.loadtxt("JSD/{}_{}_jsd.txt".format(i, amp))
        jsd_mean.append(np.mean(jsdarr))
        jsd_error.append(np.std(jsdarr))

        plastarr = np.loadtxt("PLAST/{}_{}_plast.txt".format(i, amp))
        plast_mean.append(np.mean(plastarr))
        plast_error.append(np.std(plastarr))

    plt.figure(1)
    plt.plot(amplifications, plast_mean, c = colours[colourcount]) # , marker = markers[markercount]
    plt.errorbar(amplifications, plast_mean, yerr = plast_error, c = colours[colourcount], fmt = 'o', markersize = 0, barsabove = True, capsize = 3)

    plt.figure(2)
    plt.plot(amplifications, jsd_mean, c = colours[colourcount]) # , marker = markers[markercount]
    plt.errorbar(amplifications, jsd_mean, yerr = jsd_error, c = colours[colourcount], fmt = 'o', markersize = 0, barsabove = True, capsize = 3)

    colourcount = (colourcount + 1) % len(colours)
  
plt.figure(1)
plt.xlabel("Multiplication factor for maximum range" , c='r')
plt.ylabel("Plasticity", c='r')
plt.title("All parameter ranges varied", c='r')
plt.legend(topofiles)
plt.savefig("paramvar_plast.png")

plt.figure(2)
plt.xlabel("Multiplication factor for maximum range" , c='r')
plt.ylabel("JSD from WT", c='r')
plt.title("All parameter ranges varied", c='r')
plt.legend(topofiles)
plt.savefig("paramvar_jsd.png")
