#CSI FID In Vivo: My Scan
import suspect
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits import mplot3d
import spect_functions as sfun
import json


global_first_phase= -5.89e-3
zero_phase= np.pi

data= suspect.io.load_twix("/Users/johnnyderhovagimian/Desktop/31P_Data_24Channel/Johnny_Oct20/meas_MID00089_FID06718_csi_fid_40.dat") #8x10x8x8x24x2048

#Step 1: Create noise matrix for data whitening

data_noise= data[:,:,:,:,:,-200:]
data_noise= np.moveaxis(data_noise, 4,0).reshape(24,-1)
data_noise= data_noise[:,data_noise[0]!= 0]

#Step 2: Whiten data and average results
data_averaged= np.mean(data,axis=1)
whitened_data= suspect.processing.channel_combination.whiten(data_averaged, data_noise)

#Step 3: Perform spatial FT
spatial_FT= np.zeros((8,8,8,24,whitened_data.np), 'complex')
spatial_FT= np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(whitened_data, axes= (0,1,2)), axes=(0,1,2)), axes=(0,1,2))
spatial_FT= data.inherit(spatial_FT)

#Step 4: Channel combination
cc_wfp= spatial_FT.inherit(np.zeros((8,8,8,data.np), 'complex'))
for i in range(8):
    for j in range(8):
        for k in range(8):
            cc_wfp[i,j,k,0:2048]=sfun.phase_and_scale(sfun.combine(spatial_FT[i,j,k], sfun.first(spatial_FT[i,j,k]), global_first_phase, zero_phase))

cc_wfp= spatial_FT.inherit(cc_wfp)
cc_wfp_filtered= cc_wfp[:,:,:]*sfun.apod(data,60)

#Step 5: Apply FT along FIDs
spectrum_cc_wfp= np.fft.fftshift(np.fft.fft(cc_wfp, axis=-1), axes=-1)
spectrum_cc_wfp= cc_wfp.inherit(spectrum_cc_wfp)

#Plot spectroscopic image along axial slice
fig1, ax= plt.subplots(8,8, figsize=(12,12), sharey= True)
fig1.subplots_adjust(hspace=0, wspace=0)
fig1.suptitle('31P In Vivo CSI Image: Axial Slice', weight='bold')
for i in range(8):
    for j in range(8):
        ax[i,j].plot(np.real(np.real(cc_wfp_filtered[i,4,j].spectrum())))
        ax[i,j].set_axis_off()

fig2= plt.figure()
plt.plot(data.frequency_axis_ppm(),np.real(cc_wfp_filtered[4,4,4,:].spectrum()))
plt.xlim([30,-30])

# #Step 6: Data Fitting with AMARES
filename= "/Users/johnnyderhovagimian/Desktop/Python/31P_Spectroscopy/singlet_model.json"
with open(filename) as fin:
    singlet_model = json.load(fin)
# singlet_model= suspect.fitting.singlet.Model.load(filename)

fits={}
for i in range(8):
    for j in range(8):
        fits[i,j]= suspect.fitting.singlet.fit(cc_wfp[i,4,j,:], singlet_model)


pcr_map=np.zeros((8,8))
for i in range(8):
    for j in range(8):
        pcr_map[i,j]= fits[i,j]["model"]["pcr"]["amplitude"]


pcr_fwhm= np.zeros(6)
count=0
#PCr fwhm average across 6 central voxels
for i in np.arange(4,6):
    for j in np.arange(3,6):
        pcr_fwhm[count]= fits[i,j]['model']['pcr']['fwhm']
        count += 1

pcr_fwhm_ave= np.mean(pcr_fwhm)
pcr_fwhm_std= np.std(pcr_fwhm)



pcr_max= np.max(pcr_map)

pcr_map /= pcr_max
fig3= plt.figure()
plt.imshow(pcr_map[1:,1:])
cbar3= plt.colorbar()
plt.title('PCr SNR Map: Axial Slice', weight= 'bold')
cbar3.set_label('SNR (A.U.)')





plt.show()