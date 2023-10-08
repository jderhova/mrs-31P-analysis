import suspect
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import spect_functions as sfun
import pandas as pd

global_first_phase= 0.00885
zero_phase= 1.8

data= suspect.io.load_twix('/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31PData_PrototypeCoil/20211103/meas_MID00155_FID05901_csi_fid.dat')

data_averaged= np.mean(data,axis=1)

spatial_FT= np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data_averaged, axes=(0,1,2)), axes=(0,1,2)), axes=(0,1,2))
spatial_FT= data.inherit(spatial_FT)

spatial_FT_filtered= spatial_FT[:,:,:]*sfun.apod(data,40)

#frequency correction: to place PCr at 0ppm
for i in range(8):
    for j in range(8):
        for k in range(8):
            FID= spatial_FT[i,k,k]
            FID_filtered= spatial_FT_filtered[i,j,k]
            df= suspect.processing.frequency_correction.residual_water_alignment(FID*sfun.apod(data,10))
            df_filtered= suspect.processing.frequency_correction.residual_water_alignment(FID_filtered*sfun.apod(data,10))

            spatial_FT[i,j,k]= FID.adjust_frequency(-df)
            spatial_FT_filtered[i,j,k]= FID_filtered.adjust_frequency(-df_filtered)

#Zero-order and first-order phase correction
for i in range(8):
    for j in range(8):
        for k in range(8):
            spatial_FT[i,j,k]= spatial_FT[i,j,k].adjust_phase(zero_phase, global_first_phase,0)
            # spatial_FT_filtered[i,j,k]= spatial_FT_filtered[i,j,k].adjust_phase(zero_phase, global_first_phase,0)


spectrum= np.fft.fftshift(np.fft.fft(spatial_FT, axis=-1), axes=-1)
spectrum= spatial_FT.inherit(spectrum)

spectrum_filtered= np.fft.fftshift(np.fft.fft(spatial_FT_filtered, axis=-1), axes=-1)
spectrum_filtered= spectrum.inherit(spectrum_filtered)

#Plot Spectrum in central Axial Slice
fig1, ax1= plt.subplots(8,8, sharey=True)
fig1.subplots_adjust(hspace=0, wspace=0)
fig1.suptitle('31P In Vivo CSI: Axial Slice', weight='bold')
for i in range(8):
    for j in range(8):
        ax1[i,j].plot(np.abs(spectrum_filtered[i,3,j]))
        ax1[i,j].set_axis_off()

#Plot single spectrum
fig2= plt.figure()
plt.plot(data.frequency_axis_ppm(), np.abs(spectrum_filtered[6,3,3]))
plt.xlim([30,-30])
plt.xlabel('frequency (ppm)')
plt.ylabel('Intensity [A.U.]')

#SNR plot for PCr
snr_map= np.zeros((8,8,8))
for i in range(8):
    for j in range(8):
        for k in range(8):
            real_spectrum= np.real(spectrum[i,j,k])
            snr_map[i,j,k]= np.sum(real_spectrum[1000:1048])/np.std(real_spectrum[-200:])

            if (snr_map[i,j,k]<0):
                snr_map[i,j,k]= 0


snr_map /= np.max(snr_map)

fig3= plt.figure()
snr_map= plt.imshow(snr_map[:,3,:])
plt.title("SNR Map of PCr in Central Axial Slice", weight='bold')
fig3.colorbar(snr_map)


plt.show()