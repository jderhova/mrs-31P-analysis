#Noise Correlation Matrix obtained from Noise scan of my brain
import suspect
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits import mplot3d
import spect_functions as sfun
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

data= suspect.io.load_twix("/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31P_Data_24Channel/Johnny_Oct20/meas_MID00087_FID06716_fid_noise.dat")

#Step 1: Create noise matrix, each row corresponds to a different channel
noise= data[:,:,:]
noise= np.moveaxis(noise,1,0).reshape(24,-1)

#Step 2: Noise correlation matrix
noise_corr_matrix= np.corrcoef(noise)
fig5= plt.figure()

colors= ["black", "red", "orange", "yellow"]
cmap= LinearSegmentedColormap.from_list('Custom', colors, N=512)
plt.imshow(np.abs(noise_corr_matrix), cmap=cmap)
plt.xticks(np.linspace(0,23,24), np.arange(1,25), fontsize= 8)
plt.yticks(np.linspace(0,23,24), np.arange(1,25), fontsize= 8)
plt.xlabel('Channel', fontsize=12)
plt.ylabel('Channel', fontsize=12)
cbar= plt.colorbar(ticks= np.linspace(0,1,11))
cbar.set_label('Correlation', fontsize=12)
plt.title('Noise Correlation Matrix', fontsize=12, weight= 'bold')
plt.clim(vmin=0, vmax=1) 


#Determine mean and standard deviations
corr_values= np.zeros(552, 'complex')
count= 0
for i in range(24):
    for j in range(24):
        if (i != j):
            corr_values[count]= noise_corr_matrix[i][j]
            count += 1


mean_corr= np.mean(np.abs(corr_values))
std_corr= np.std(np.abs(corr_values))

print(mean_corr)
print(std_corr)



plt.show()

#Worst case noise correlation is now 53% 