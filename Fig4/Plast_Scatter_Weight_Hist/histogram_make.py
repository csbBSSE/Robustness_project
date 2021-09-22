

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '25', 'axes.titlesize':'25', 'axes.labelsize':'25', 'xtick.labelsize':'20', 'ytick.labelsize':'20'})


arr1 = np.loadtxt("weight.txt")
arr1 *=-1

a = -10
b = +10

arr2 = np.clip(arr1, a , b)
posarr = np.linspace(-10,10,9)
print(posarr)
xticks = list(posarr)
for i in range (len(xticks)):
    xticks[i] = str(xticks[i])

c = len(xticks)
xticks[0] = " ≤{}".format(a)
xticks[c-1] = "≥{}".format(b)

fig,ax = plt.subplots()

    


plt.hist(arr2, bins = 20)
plt.xticks(posarr,xticks)

plt.axvline(np.median(arr1), color='r', linestyle='dashed', linewidth=4)
plt.title("Weights for Plasticity", fontsize = 30)
plt.ylabel("Number of networks", fontsize = 30)
plt.xlabel("Optimal weight for PFL" , fontsize = 30)
r = 2
f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)   


plt.tight_layout()
median = np.median(arr1)
textstr = r'$\mathrm{Median}=%.2f$' % (median, )
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.10, 0.95, textstr, transform=ax.transAxes, fontsize=25,
        verticalalignment='top', bbox=props)


plt.savefig("weighthist.jpg", transparent = True)

plt.clf()
print(np.median(arr1))