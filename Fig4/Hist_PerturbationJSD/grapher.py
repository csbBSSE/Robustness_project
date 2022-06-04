import numpy as np
import matplotlib.pyplot as plt
from math import pow
import os
from scipy.stats import norm, zscore
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from matplotlib import rcParams
import pandas as pd
from scipy.stats import pearsonr
import os
import time
from os import listdir
from os.path import isfile, join




topofiles= [os.path.splitext(f)[0] for f in listdir("topofiles/") if isfile(join("topofiles/", f))]



for w in range(len(topofiles)):

    version='cnt2'

    racjsd=np.loadtxt("racipe_{}_jsd.txt".format(topofiles[w]))
    booljsd=np.loadtxt("{}_{}_jsd.txt".format(version,topofiles[w]))


    plt.scatter (booljsd,racjsd)

    corr, _ = pearsonr(racjsd,booljsd)
    plt.title("Pearson coef= {:.6f}".format(corr) )
    plt.savefig("{}plots/jsd_{}_{}.jpg".format(version,version,topofiles[w]))
    plt.clf()