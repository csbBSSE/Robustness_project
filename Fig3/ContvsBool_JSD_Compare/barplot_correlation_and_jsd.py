import numpy as np
import matplotlib.pyplot as plt

import matplotlib
 
matplotlib.rcParams.update({'font.weight':'bold', 'xtick.color':'0.3', 'ytick.color':'0.3', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'figure.titleweight':'bold', 'text.color':'0.3', 'axes.labelcolor':'0.3', 'axes.titlecolor':'0.3', 'font.size': '25', 'axes.titlesize':'50', 'axes.labelsize':'35', 'xtick.labelsize':'33', 'ytick.labelsize':'30', 'legend.fontsize':'33'})
 

xarrno = np.array([0,1,2,3,4])

xarr4=['GRHL2', 'GRHL2wa', 'OCT4', 'OVOL', 'OVOLsi']
from matplotlib.pyplot import figure

yarr41=[7.644880313883776202e-01,
3.069287309965897381e-01,
5.126528889437401704e-01,
4.454826171985344363e-01,
5.221020555981911171e-01
]
yarr42=[
8.823399278640420285e-01,
7.183037669715701679e-01,
9.096483911460175875e-01,
9.119248553619956343e-01,
9.133876733701814521e-01
]

r = 2
fig = plt.figure()

plt.bar(xarrno,yarr41, width = 0.4 ,color='r')
plt.xticks(xarrno + 0.2 , xarr4 , fontweight="bold" , c='0.3' )
plt.bar(xarrno+0.4,yarr42, width = 0.4, color='b')
plt.xticks(rotation=15)
plt.ylabel("JSD Correlation with Racipe", fontweight="bold" , c='0.3' )
#plt.xlabel("Networks" ,fontweight="bold" , c='0.3' )
#plt.title("Boolean vs Cont. Correlation with Racipe")

legend = plt.legend(['Boolean', 'Continuous'] )
plt.setp(legend.get_texts(), color='0.3', fontweight="bold" )

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off

f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)
axes = plt.gca()
axes.set_ylim([0,1.3])
plt.tight_layout()
plt.savefig("correlation.png", transparent = True)
plt.close()

######################

import numpy as np
import matplotlib.pyplot as plt
xarrno = np.array([0,1,2,3,4])
import matplotlib
xarr4=['GRHL2','GRHL2wa', 'NRF2', 'OCT4', 'OVOL']


yarr41=[0.2666,
0.3314,
0.4508,
0.2803,
0.3527  ,
]
yarr42=[
0.1389,
0.1907,
0.1942,
0.1394,
0.1456
]

r = 2
fig = plt.figure()


plt.bar(xarrno,yarr41, width = 0.4 ,color='r')
plt.xticks(xarrno + 0.2 , xarr4,  fontweight="bold" , c='0.3')
plt.bar(xarrno+0.4,yarr42, width = 0.4, color='b')

plt.ylabel("JSD from RACIPE",  fontweight="bold" , c='0.3')
#plt.xlabel("Networks", fontweight="bold" , c='0.3')
plt.xticks(rotation=15)
#plt.title("Boolean vs Cont. JSD")
legend = plt.legend(['Boolean', 'Continuous'] )
plt.setp(legend.get_texts(), color='0.3', fontweight="bold" )

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off

f=r*np.array(plt.rcParams["figure.figsize"])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(f)
plt.tight_layout()

plt.savefig("jsd.png" , transparent = True)









