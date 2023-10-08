#12x12x8 CSI Acquired In Vivo
#Subject: Christian
#Coil: 31P Birdcage/24-channel receive

import suspect
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits import mplot3d
import json
import pickle
import spect_functions as sfun 

# First-order and zero-order phase correction parameters to change
first_order_phase= 0.0064#0.005
zero_order_phase= 0.06#0.06

data= suspect.io.load_twix("/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31P_Data_24Channel/Feb_17/meas_MID00043_FID16113_csi_fid_12x12_sag_fov240.dat")

#Step 1: Create noise matrix for data whitening
data_noise= data[:,:,:,:,:,-200:]
data_noise= np.moveaxis(data_noise, 4,0).reshape(24,-1)
data_noise= data_noise[:,data_noise[0]!= 0]

#Step 2: Whiten Data to remove noise correlations
data_averaged= np.mean(data,axis=1)
whitened_data= suspect.processing.channel_combination.whiten(data_averaged, data_noise)

#Step 3: Perform spatial FT
spatial_FT= np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(whitened_data, axes= (0,1,2)), axes=(0,1,2)), axes=(0,1,2))
spatial_FT= data.inherit(spatial_FT)

#Step 4: Channel combination

#Weighted by First Point
cc_wfp= spatial_FT.inherit(np.zeros((12,8,12,data.np), 'complex'))
for i in range(12):
    for j in range(8):
        for k in range(12):
            cc_wfp[i,j,k]=sfun.phase_and_scale(sfun.combine(spatial_FT[i,j,k], sfun.first(spatial_FT[i,j,k])), zero_order_phase, first_order_phase)
            
cc_wfp_filtered= cc_wfp[:,:,:]*sfun.apod(data,40)

#Step 5: Apply FT along FIDs
spectrum_cc_wfp= np.fft.fftshift(np.fft.fft(cc_wfp, axis=-1), axes=-1)
spectrum_cc_wfp= cc_wfp.inherit(spectrum_cc_wfp)

central_sagittal_filtered= np.flip(cc_wfp_filtered[:,3,:],0)
central_sagittal= np.flip(cc_wfp[:,3,:],0)

# Plot spectroscopic image along axial slice
# fig1, ax= plt.subplots(12,12, figsize=(12,12), sharey= True)
# fig1.subplots_adjust(hspace=0, wspace=0)
# fig1.suptitle('31P In Vivo CSI Image: Sagittal Slice', weight='bold')
# for i in range(12):
#     for j in range(12):
#         ax[i,j].plot(np.abs(central_sagittal_filtered[i,j].spectrum()))
#         ax[i,j].set_axis_off()


#Plot single spectrum from central sagittal CSI grid, this provides a single spectrum that can be observed to assess the quality of phase corrections
# fig3= plt.figure()
# plt.plot(np.abs(central_sagittal_filtered[7,8].spectrum()), color= "green")
# # plt.xlim([15,-15])
# plt.xlabel('ppm', fontsize= 12)
# plt.ylabel('Intensity (A.U.)', fontsize=12)

#Calculate PCr SNR
# PCr_SNR= np.zeros([12,8,12], 'complex')

# for i in range (12):
#     for j in range (8):
#         for k in range (12):
#             PCr_SNR[i,j,k]= np.sum(np.abs(cc_wfp_filtered.spectrum()[i,j,k,980:1068]))

# PCr_SNR= np.abs(PCr_SNR)

# fig3=plt.figure()
# snr_plot=plt.imshow(np.flip(PCr_SNR[:,4,:],0))
# cbar1= plt.colorbar(snr_plot)
# cbar1.set_label('SNR (A.U.)')
# plt.title('SNR Map: Axial Slice', weight= 'bold')


# #Step 6: Data Fitting with AMARES
filename= "/Users/johnnyderhovagimian/Desktop/Python/31P_Spectroscopy/singlet_model.json"
with open(filename) as fin:
    singlet_model = json.load(fin)


##__ Single Spectrum Fit____
# fit_test= suspect.fitting.singlet.fit(central_sagittal_filtered[8,8], singlet_model)

# plt.figure()
# plt.plot(np.real(np.fft.fftshift(np.fft.fft(fit_test['fit']))))
# plt.plot(np.real(central_sagittal_filtered[8,8].spectrum()))


## Fit spectra and save fits to a file. Do this once to avoid rerunning AMARES fit each time.
fits={}
for i in range(12):
    for j in range(12):
        fits[i,j]= suspect.fitting.singlet.fit(central_sagittal[i,j], singlet_model)

#Save dictionary to file- do this once to avoid time consuming AMARES fitting
with open('fits.pkl', 'wb') as fp:
    pickle.dump(fits, fp)
    print('fits successfully saved to file')

# #Plot spectroscopic image along sagittal slice
fig7, ax= plt.subplots(12,12, figsize=(12,12), sharey= True)
fig7.subplots_adjust(hspace=0, wspace=0)
fig7.suptitle('31P In Vivo CSI Image: Sagittal Slice', weight='bold')
for i in range(12):
    for j in range(12):
        ax[i,j].plot(np.abs(central_sagittal_filtered[i,j].spectrum()))
        ax[i,j].plot(np.real(np.fft.fftshift(np.fft.fft(fits[i,j]['fit']))))
        ax[i,j].set_axis_off()

#PCr SNR Map
pcr_snr_central_sag= np.zeros((12,12), 'complex')

for i in range(12):
    for j in range(12):
        pcr_snr_central_sag[i,j]= fits[i,j]['model']['pcr']['amplitude']


#Alpha-ATP SNR Map
atpa_snr_central_sag= np.zeros((12,12), 'complex')

for i in range(12):
    for j in range(12):
        atpa_snr_central_sag[i,j]=fits[i,j]['model']['atpa']['amplitude']
        if(atpa_snr_central_sag[i,j]==0): #Prevent divide by zero error when calculating PCr/ATP map
            atpa_snr_central_sag[i,j]= 1

#Gamma-ATP SNR Map
atpc_snr_central_sag= np.zeros((12,12), 'complex')

for i in range(12):
    for j in range(12):
        atpc_snr_central_sag[i,j]=fits[i,j]['model']['atpc']['amplitude']
        if (atpc_snr_central_sag[i,j]==0):#Prevent divide by zero error when calculating PCr/ATP map
            atpc_snr_central_sag[i,j]=1



#Create a sagittal mask using PCr SNR values
sagittal_mask= np.zeros((12,12))
for i in range (12):
    for j in range(12):
        if(pcr_snr_central_sag[i,j] > 5e+3):
            sagittal_mask[i,j]= 1


#Calculate PCR/Mean ATP ratio and apply mask
mean_ATP= np.add(atpa_snr_central_sag,atpc_snr_central_sag)/2

PCR_ATP_map= np.abs(pcr_snr_central_sag/mean_ATP)

PCR_ATP_map= np.multiply(PCR_ATP_map, sagittal_mask)

plt.figure()
pcr_snr=plt.imshow(np.abs(pcr_snr_central_sag))
cbar1= plt.colorbar(pcr_snr)
plt.title('PCr Plot')

plt.figure()
m_atp=plt.imshow(np.abs(mean_ATP))
plt.title('Mean ATP Plot')
#cbar2= plt.colorbar(pcr_atp)

plt.figure()
plt.imshow(np.abs(atpc_snr_central_sag))
plt.title('Gamma-ATP Plot')

plt.figure()
plt.imshow(abs(atpa_snr_central_sag))
plt.title('Alpha-ATP Plot')

plt.figure()
pcr_atp=plt.imshow(PCR_ATP_map, vmax= 15)
cbar5= plt.colorbar(pcr_atp)
plt.title('PCr/Mean ATP')

plt.show()

