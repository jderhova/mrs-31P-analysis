#In Vivo CSI Acquisition: 8x8x8
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


data= suspect.io.load_twix("/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31P_Data_24Channel/Johnny_Oct25/20221025/meas_MID00068_FID07321_csi_fid.dat")

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
cc_wfp= spatial_FT.inherit(np.zeros((8,8,8,data.np), 'complex'))
for i in range(8):
    for j in range(8):
        for k in range(8):
            cc_wfp[i,j,k]=sfun.phase_and_scale(sfun.combine(spatial_FT[i,j,k], sfun.first(spatial_FT[i,j,k]), first_order_phase, zero_order_phase))
            
cc_wfp_filtered= cc_wfp[:,:,:]*sfun.apod(data,40)

#SVD
cc_svd= spatial_FT.inherit(np.zeros((8,8,8,data.np),'complex'))
for i in range(8):
    for j in range(8):
        for k in range(8):
            cc_svd[i,j,k]= sfun.phase_and_scale(sfun.combine(spatial_FT[i,j,k], suspect.processing.channel_combination.svd_weighting(spatial_FT[i,j,k]), 
            first_order_phase, zero_order_phase))

#apodization with exponential filter
cc_svd_filtered= cc_svd[:,:,:]*sfun.apod(data,40)

#Step 5: Apply FT along FIDs
spectrum_cc_wfp= np.fft.fftshift(np.fft.fft(cc_wfp, axis=-1), axes=-1)
spectrum_cc_wfp= cc_wfp.inherit(spectrum_cc_wfp)

spectrum_cc_svd= np.fft.fftshift(np.fft.fft(cc_svd, axis=-1), axes=-1)
spectrum_cc_svd= cc_svd.inherit(spectrum_cc_svd)

central_sagittal_filtered= np.flip(cc_wfp_filtered[1:7,0:6,3],0)
central_sagittal= np.flip(cc_wfp[1:7,0:6,3],0)

#Plot spectroscopic image along axial slice
# fig1, ax= plt.subplots(6,6, figsize=(12,12), sharey= True)
# fig1.subplots_adjust(hspace=0, wspace=0)
# fig1.suptitle('31P In Vivo CSI Image: Axial Slice', weight='bold')
# for i in range(6):
#     for j in range(6):
#         ax[i,j].plot(np.abs(central_sagittal[i,j].spectrum()))
#         ax[i,j].set_axis_off()


#Plot Central Voxel, peripheral voxel, and voxel in cerebellum from Central sagitall slice

#Central Voxel
# fig2= plt.figure()
# plt.plot(data.frequency_axis_ppm(), np.abs(central_sagittal[2,3].spectrum()), color="red")
# plt.xlim([15,-15])
# plt.xlabel('Frequency [ppm]', fontsize=12)
# plt.ylabel('Intensity [A.U.]', fontsize=12)

#Vocel in Cerebellum
# fig3= plt.figure()
# plt.plot(data.frequency_axis_ppm(), np.abs(central_sagittal[4,4].spectrum()), color="green")
# plt.xlim([15,-15])
# plt.xlabel('Frequency [ppm]', fontsize=12)
# plt.ylabel('Intensity [A.U.]', fontsize=12)


#Plot single spectrum from central sagittal CSI grid, this provides a single spectrum that can be observed to assess the quality of phase corrections
fig3= plt.figure()
plt.plot(np.real(cc_wfp_filtered[3,4,3].spectrum()))
# plt.xlim([30,-30])
plt.xlabel('ppm', fontsize= 12)
plt.ylabel('Intensity (A.U.)', fontsize=12)


# #Step 6: Data Fitting with AMARES
filename= "/Users/johnnyderhovagimian/Desktop/Python/31P_Spectroscopy/singlet_model.json"
with open(filename) as fin:
    singlet_model = json.load(fin)

fits={}
for i in range(6):
    for j in range(6):
        fits[i,j]= suspect.fitting.singlet.fit(central_sagittal[i,j], singlet_model)

#Plot spectroscopic image along axial slice
fig7, ax= plt.subplots(6,6, figsize=(12,12), sharey= True)
fig7.subplots_adjust(hspace=0, wspace=0)
fig7.suptitle('31P In Vivo CSI Image: Axial Slice', weight='bold')
central_sagittal= np.flip(cc_wfp_filtered[1:7,0:6,3],0)
for i in range(6):
    for j in range(6):
        ax[i,j].plot(np.abs(central_sagittal[i,j].spectrum()))
        ax[i,j].plot(np.real(np.fft.fftshift(np.fft.fft(fits[i,j]['fit']))))
        ax[i,j].set_axis_off()


#plot fitted data
# fig4,ax4= plt.subplots(6,6, sharey=True)
# fig4.suptitle('Amares Fitting')
# for i in range(6):
#     for j in range(6):
#         ax4[i,j].plot(np.fft.fftshift(np.fft.fft(fits[i,j]['fit'])).real)

sagittal_mask= np.array([[0, 1, 1, 1, 1, 0],[1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1],[1, 1, 1, 1, 1, 1],[0, 0, 0, 1, 1, 1],[0, 0, 0, 1, 1, 1]], np.int32)

#Pcr map with fitted data
pcr_map1= np.zeros((6,6), 'complex')
for i in range(6):
    for j in range(6):
        pcr_map1[i,j]= fits[i,j]['model']['pcr']['amplitude']

# pcr_map1 /= np.amax(pcr_map1)

#Calculate PCr SNR map as area under real valued spectrum
pcr_map2= np.zeros((6,6), 'complex')
for i in range(6):
    for j in range(6):
        pcr_map2[i,j]= np.sum(np.real(central_sagittal_filtered[i,j].spectrum())[960:1080])

# pcr_map2 /= np.amax(pcr_map2) #normalize

# fig5= plt.figure()
# snr_plot=plt.imshow(np.abs(pcr_map2))
# cbar3= plt.colorbar(snr_plot)
# cbar3.set_label('SNR (A.U.)')
# plt.title('PCr SNR Map: Central Sagittal Slice', weight= 'bold')

#Plot PCr SNR map
fig6= plt.figure()
snr_plot2= plt.imshow(np.abs(pcr_map2))
cbar4= plt.colorbar(snr_plot2)
cbar4.set_label('SNR (A.U)')
plt.title('PCr SNR Map', weight='bold')

#gamma-ATP SNR map
atpc_map= np.zeros((6,6), 'complex')
for i in range(6):
    for j in range(6):
        atpc_map[i,j]= fits[i,j]['model']['atpc']['amplitude'] 
        if (atpc_map[i,j]== 0):
            atpc_map[i,j]= 1
        # gATP_map[i,j]= np.sum(np.abs(central_sagittal_filtered[i,j].spectrum())[1142:1226])

# gATP_pcr= np.multiply(pcr_map1/gATP_map, sagittal_mask)

# print(np.mean(gATP_pcr))

atpa_map= np.zeros ((6,6), 'complex')
for i in range(6):
    for j in range(6):
        atpa_map[i,j]= fits[i,j]['model']['atpa']['amplitude']
        if (atpa_map[i,j]==0):
            atpa_map[i,j]= 1

mean_atp= np.add(atpa_map, atpc_map)/2

pcr_atp_map= np.multiply(pcr_map1/mean_atp, sagittal_mask)

fig6= plt.figure()
snr_plot3= plt.imshow(np.abs(atpc_map))
cbar3= plt.colorbar(snr_plot3)
cbar3.set_label('SNR (A.U.)')
plt.title('gamma-ATP SNR Map', weight= 'bold')

fig7= plt.figure()
snr_plot4= plt.imshow(np.abs(atpa_map))
cbar4= plt.colorbar(snr_plot4)
cbar4.set_label('SNR (A.U.)')
plt.title('alpha-ATP SNR Map', weight= 'bold')

fig8= plt.figure()
snr_plot5= plt.imshow(np.abs(pcr_atp_map))
cbar5= plt.colorbar(snr_plot5)
cbar5.set_label('SNR (A.U.)')
plt.title('PCr/ATP Map', weight='bold')


#Calculate Mean and Standed deviation of PCr/ATP
mean_pcr_atp= np.mean(pcr_atp_map[pcr_atp_map !=0])
std_pcr_atp= np.std(pcr_atp_map[pcr_atp_map != 0])
print(f'Mean PCr/ATP is {mean_pcr_atp} and std is {std_pcr_atp}')


# fig7= plt.figure()
# snr_plot4= plt.imshow(np.abs(gATP_pcr), vmin=0, vmax=2)
# cbar4= plt.colorbar(snr_plot4)
# cbar4.set_label('SNR (A.U.)')
# plt.title('PCr/gATP SNR Map')

#### Illustrate coil combination for 1 voxel
# uncombined_FID= spatial_FT[4,4,4]*sfun.apod(data,40)
# combined_FID= cc_wfp[4,4,4]*sfun.apod(data,40)

# fig7,ax7= plt.subplots(2,1)
# fig7.subplots_adjust(hspace= 0.5)
# ax7[0].set_title('Single Voxel Uncombined FIDs')
# ax7[0].set_xlabel('Time (s)')
# ax7[0].set_ylabel('Amplitude [A.U.]')

# ax7[1].set_title('Single Voxel Combined FID')
# ax7[1].set_xlabel('Time (s)')
# ax7[1].set_ylabel('Amplitude [A.U.]')


plt.show()

