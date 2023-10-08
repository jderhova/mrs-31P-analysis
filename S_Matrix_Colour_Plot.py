# S-matrix as a colour plot
# Rx coil loaded with human head

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

df= pd.read_excel(r'SMatrix_Human_Head.xlsx', header=None) #import excel table

s_matrix= df.to_numpy() #convert to numpy matrix

colors= ["black", "red", "orange", "yellow"]
cmap= LinearSegmentedColormap.from_list('Custom', colors, N=512)
sMatrixPlot=plt.imshow(s_matrix, cmap= cmap,vmin=-30, vmax=-7)
plt.xticks(np.linspace(0,23,24), np.arange(1,25), fontsize= 8)
plt.yticks(np.linspace(0,23,24), np.arange(1,25), fontsize= 8)
ticks= np.arange(-7,-30, -2)
cbar= plt.colorbar(sMatrixPlot, ticks=ticks)
cbar.set_label('S-Parameter [dB]', weight='bold')
plt.title('S-Parameter Matrix', weight= "bold")



plt.show()