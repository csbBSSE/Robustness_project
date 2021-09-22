import matplotlib.pyplot as plt
from matplotlib import rcParams as r
import numpy as np
import matplotlib
from matplotlib import rcParams

bool = np.loadtxt("booltime.txt")
rac = np.loadtxt("ractime.txt")

r.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size':'30'})

fig, ax = plt.subplots()
ax.set_yscale('log')
x = [4,5,6,7,8,9,10]
plt.plot(x, bool[:, 0], label = "Continuous", c = 'b')
plt.errorbar(x, bool[:, 0], bool[:, 1], capsize = 5, c = 'b' , linewidth = 5, elinewidth = 2, ecolor = 'k')
plt.plot(x, rac[:, 0], label = "RACIPE", c = 'r')
plt.errorbar(x, rac[:, 0], rac[:, 1], capsize = 5, c = 'r', linewidth = 5 , elinewidth = 2 , ecolor = 'k')

plt.xlabel("Network Size")
plt.ylabel("Computational Time")
leg = ax.legend()

for line in leg.get_lines():
    line.set_linewidth(4.0)
    
plt.title("RACIPE vs Continuous: Computation Time" , fontsize = 28)
f=2*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)
plt.savefig("comptime.png")
