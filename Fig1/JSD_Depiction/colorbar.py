import matplotlib.pyplot as plt
import matplotlib as mpl

fig = plt.figure()
ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])

cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap = plt.get_cmap('Blues'))

plt.xticks(fontsize = 15)
plt.title("JSD", y = 1.5, fontsize = 40)
plt.savefig('just_colorbar', bbox_inches='tight')