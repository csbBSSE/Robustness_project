from scipy.spatial.distance import jensenshannon
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 500
size = 1000
stdev = 0.3
a = np.random.normal(-1, stdev, size)
b = np.random.normal(1, stdev, size)
x1 = np.linspace(-3,0,size)
x2 = np.linspace(0,3,size)

f, (ax2, ax1) = plt.subplots(1, 2, sharey=True, figsize = (10,4))

ax1.plot(x1, norm.pdf(x1, -1, stdev), c = 'b', linewidth = 5)
ax1.plot(x2, norm.pdf(x2, 1, stdev), c = 'r', linewidth = 5)
ax1.set_ylim([0.01,1.5])
ax1.set_xlim([-2,2])
ax1.set_xticks([])
ax1.set_yticks([])
#ax1.savefig("jsd1.png")

#plt.clf()

ax2.plot(x1, norm.pdf(x1, -1, stdev), c = 'r', linewidth = 5)
ax2.plot(x1, norm.pdf(x1, -1.08, stdev), c = 'b', linewidth = 5)
ax2.set_ylim([0.01,1.5])
ax2.set_xlim([-2,0])
ax2.set_xticks([])
ax2.set_yticks([])
#plt.savefig("jsd2.png")
f.tight_layout()
f.savefig("jsd.png")
