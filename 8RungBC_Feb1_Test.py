import suspect
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import spect_functions as sfun
import pandas as pd

global_first_phase= 0
zero_phase= 0

data= suspect.io.load_twix('/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31PData_PrototypeCoil/Feb_1/meas_MID00065_FID14966_csi_fid_8rung.dat') 

#Average data across to acquisitions
data= np.mean(data, axis=1)


plt.figure()
plt.plot(np.real(data[10,3,10,:]))

#Apply spatial fourier transform to transform to image space
spatial_FT= np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data, axes=(0,1,2)), axes=(0,1,2)), axes=(0,1,2))
spatial_FT= data.inherit(spatial_FT)

#Apodize FIDs with exponential filter, for visual diplay purposes only
spatial_FT_filtered= spatial_FT[:,:,:]*sfun.apod(data,40)

#Zero-Order Phase Adjustment 
for i in range(16):
    for j in range(8):
        for k in range(16):
            spatial_FT[i,j,k]=spatial_FT[i,j,k].adjust_phase(-global_first_phase)
            spatial_FT_filtered[i,j,k]= spatial_FT_filtered[i,j,k].adjust_phase(-global_first_phase)

#Frequency adjust: this centers the peak at 0ppm
for i in range(16):
    for j in range(8):
        for k in range(16):
            FID= spatial_FT[i,j,k]
            FID_filtered= spatial_FT_filtered[i,j,k]
            df= suspect.processing.frequency_correction.residual_water_alignment(FID*sfun.apod(data,10))
            df_filtered= suspect.processing.frequency_correction.residual_water_alignment(FID_filtered*sfun.apod(data,10))

            spatial_FT[i,j,k]= spatial_FT[i,j,k].adjust_frequency(-df)
            spatial_FT_filtered[i,j,k]= spatial_FT_filtered[i,j,k].adjust_frequency(-df_filtered)


#FT original FIDS and apodized FIDS
spectrum= np.fft.fftshift(np.fft.fft(spatial_FT, axis=-1), axes=-1)
spectrum= spatial_FT.inherit(spectrum)

spectrum_filtered= np.fft.fftshift(np.fft.fft(spatial_FT_filtered, axis=-1), axes=-1)
spectrum_filtered= spatial_FT.inherit(spectrum_filtered)

#Plot Axial Slice
fig1, ax1= plt.subplots(16,16, sharey=True)
fig1.subplots_adjust(hspace=0, wspace=0)
fig1.suptitle('31P Phantom CSI: Axial Slice', weight='bold')
for i in range(16):
    for j in range(16):
        ax1[i,j].plot(np.real(spectrum_filtered[i,4,j]))
        ax1[i,j].set_axis_off()


#Plot Single Spectrum from Axial Slice
fig2= plt.figure()
plt.plot(np.real(spectrum_filtered[7,4,7]))
# plt.xlim([15,-15])
plt.xlabel("frequnecy [ppm]")
plt.ylabel("Intensity [A.U.]")


#Generate SNR Plot
# snr_map= np.zeros((16,8,16))
# for i in range(16):
#     for j in range(8):
#         for k in range(16):
#             snr_map[i,j,k]= np.sum(np.real(spectrum[i,j,k][480:530]))/np.std(np.real(spectrum[i,j,k][-100:]))
            
#             if (snr_map[i,j,k] < 0):
#                 snr_map[i,j,k]= 0.01

# # snr_map /= np.max(snr_map)

# #Plot SNR along axial slice
# fig3= plt.figure()
# plt.title("31P SNR Plot: Central Axial Slice", weight= 'bold')
# snr_map_axial= snr_map[:,4,:]
# snr_plot=plt.imshow(snr_map_axial)
# cbar3= plt.colorbar(snr_plot)


plt.show()
