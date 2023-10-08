#In Vivo CSI Acquisition: 8x8x8
#Subject: Johnny
#Coil: 16-rung BC/24-channel array
import suspect
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits import mplot3d
import spect_functions as sfun
import json


#### First order and Zero order phase correction parameters to change
first_order_phase= 0#0.005
zero_order_phase= 0#0.06


data= suspect.io.load_twix("/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31P_Data_24Channel/Feb_8/meas_MID00037_FID15354_csi_fid_sag.dat")

#Step 1: Create noise matrix for data whitening
data_noise= data[:,:,:,:,:,-200:]
data_noise= np.moveaxis(data_noise,4,0).reshape(24,-1)
data_noise= data_noise[:,data_noise[0]!= 0]

#Step 2: Whiten Data to remove noise correlations
data_averaged= np.mean(data,axis=1)
whitened_data= suspect.processing.channel_combination.whiten(data_averaged, data_noise)


#Step 3: Perform spatial FT
spatial_FT= np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(whitened_data, axes= (0,1,2)), axes=(0,1,2)), axes=(0,1,2))
spatial_FT= data.inherit(spatial_FT)

for i in range(24):
    plt.figure()
    plt.plot(np.abs(spatial_FT[4,4,4,i]))


#Step 4: Channel combination

#Weighted by First Point
cc_wfp= spatial_FT.inherit(np.zeros((8,8,8,data.np), 'complex'))
for i in range(8):
    for j in range(8):
        for k in range(8):
            cc_wfp[i,j,k]=sfun.phase_and_scale(sfun.combine(spatial_FT[i,j,k], sfun.first(spatial_FT[i,j,k]), first_order_phase, zero_order_phase))
            
cc_wfp_filtered= cc_wfp[:,:,:]*sfun.apod(data,40)

#Step 5: Apply FT along FIDs
spectrum_cc_wfp= np.fft.fftshift(np.fft.fft(cc_wfp, axis=-1), axes=-1)
spectrum_cc_wfp= cc_wfp.inherit(spectrum_cc_wfp)

central_sagittal_filtered= np.flip(cc_wfp_filtered[:,2,:],0)
# central_sagittal_filtered= cc_wfp_filtered[:,3,:]
central_sagittal= np.flip(cc_wfp[:,4,:],0)

#Plot spectroscopic image along axial slice
fig1, ax= plt.subplots(8,8, figsize=(12,12), sharey= True)
fig1.subplots_adjust(hspace=0, wspace=0)
fig1.suptitle('31P In Vivo CSI Image: Sagittal Slice', weight='bold')
for i in range(8):
    for j in range(8):
        ax[i,j].plot(np.abs(central_sagittal_filtered[i,j].spectrum()))
        ax[i,j].set_axis_off()


#Plot single spectrum from central sagittal CSI grid, this provides a single spectrum that can be observed to assess the quality of phase corrections
fig3= plt.figure()
plt.plot(data.frequency_axis_ppm(),np.abs(central_sagittal_filtered[2,4].spectrum()))
plt.xlim([30,-30])
plt.xlabel('ppm', fontsize= 12)
plt.ylabel('Intensity (A.U.)', fontsize=12)

plt.show()

