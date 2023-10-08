#CSI FID In Vivo: Christian's Scan
import suspect
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits import mplot3d
import spect_functions as sfun
import json



global_first_phase= 8.45e-3 #Estimated first-order phase correction
zero_phase= -0.35936 #estimated zero-order phase correction

data= suspect.io.load_twix("/Users/johnnyderhovagimian/Desktop/31P_Data_24Channel/Johnny_Oct14/meas_MID00168_FID06023_csi_fid_40.dat")# 8x10x8x8x24x2048

#Step 1: Create noise matrix for data whitening

data_noise= data[:,:,:,:,:,-100:]
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
cc_wfp_filtered= cc_wfp[:,:,:]*sfun.apod(data,40)

# cc_SV= np.pad(cc_wfp_filtered[4,4,4,:], (0,1024),'constant')
# cc_SV_FT= np.fft.fftshift(np.fft.fft(cc_SV, axis=-1),axes=-1)

# plt.plot(np.real(cc_SV_FT))

#Step 5: Apply FT along FIDs
spectrum_cc_wfp= np.fft.fftshift(np.fft.fft(cc_wfp, axis=-1), axes=-1)
spectrum_cc_wfp= cc_wfp.inherit(spectrum_cc_wfp)

#Plot spectroscopic image along axial slice
fig1, ax= plt.subplots(8,8, figsize=(12,12), sharey= True)
fig1.subplots_adjust(hspace=0, wspace=0)
fig1.suptitle('31P In Vivo CSI Image: Axial Slice', weight='bold')
for i in range(8):
    for j in range(8):
        ax[i,j].plot(np.real(cc_wfp_filtered[i,4,j].spectrum()))
        ax[i,j].set_axis_off()

fig2= plt.figure()
ppm_axis= cc_wfp.frequency_axis()/data.f0
plt.plot(data.frequency_axis_ppm(),np.real(cc_wfp_filtered[4,4,4,:].spectrum()))
plt.xlim([30,-30])
plt.xlabel('ppm', fontsize= 12)
plt.ylabel('Intensity (A.U.)', fontsize=12)

# plt.plot(data.frequency_axis_ppm(), np.imag(cc_wfp_filtered[4,4,4,:].spectrum()))


# #Step 6: Data Fitting with AMARES
filename= "/Users/johnnyderhovagimian/Desktop/Python/31P_Spectroscopy/singlet_model.json"
with open(filename) as fin:
    singlet_model = json.load(fin)
# singlet_model= suspect.fitting.singlet.Model.load(filename)

fits={}
for i in range(8):
    for j in range(8):
        fits[i,j]= suspect.fitting.singlet.fit(cc_wfp_filtered[i,4,j,:], singlet_model)

pcr_map=np.zeros((8,8))
for i in range(8):
    for j in range(8):
        pcr_map[i,j]= fits[i,j]["model"]["pcr"]["amplitude"]


pcr_fwhm= np.zeros(6)
count=0
#PCr fwhm average across 6 central voxels
for i in np.arange(3,5):
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


atpc_map= np.zeros((8,8))
for i in range(8):
    for j in range(8):
        atpc_map[i,j]= fits[i,j]["model"]["atpc"]["amplitude"]


## Create SNR maps of metabolites relative to PCr
# metabolite_maps= {}
# metabolite_list = ["pi", "atpc", "atpa", "gpetn", "petn", "pcho", "gpcho", "nad"]
# for metabolite in metabolite_list:
#     metabolite_maps[metabolite]= data.inherit(np.zeros((8,8)))
#     for i in range(8):
#         for j in range(8):
#             metabolite_maps[metabolite][i,j]= fits[i,j]["model"][metabolite]["amplitude"]/1



fig4= plt.figure()
plt.imshow(atpc_map[1:,1:])
cbar4= plt.colorbar()
# plt.cbar(vmax=3000)
plt.title('Gamma-ATP Map')


# fig4= plt.figure(3, (6.92,3.5))
# grid= ImageGrid(fig4, (0.05, 0.05, 0.9, 0.9),
#                         nrows_ncols= (2,4),
#                         axes_pad= 0.1,
#                         cbar_mode= "single",
#                         share_all= True,
#                         aspect= True)

# metabolite_titles= ["Pi", "Gamma-ATP", "Alpha-ATP", "GPE", "PE", "PCho", "GPCho", "NAD"]

# for i, metabolite in enumerate(metabolite_list):
#     imgplot= grid[i].imshow(metabolite_maps[metabolite])
#     grid.cbar_axes[i].colorbar(imgplot, ticks= [0,1])
#     grid.cbar_axes[0].set_ylabel("Ratio to phosphocreatine")
#     grid.cbar_axes[0].yaxis.labelpad=-8
#     plt.setp(grid[i].get_yticklabels(), visible=False)
#     plt.setp(grid[i].get_xticklabels(), visible=False)
#     if i < 4:
#         grid[i].set_title(metabolite_titles[i], size= "medium")
#     else:
#         grid[i].set_xlabel(metabolite_titles[i])



# noise_corr_matrix= np.corrcoef(data_noise)
# fig5= plt.figure()
# plt.imshow(np.abs(noise_corr_matrix))
# plt.xticks(np.linspace(0,23,24), fontsize= 8)
# plt.yticks(np.linspace(0,23,24), fontsize= 8)
# plt.colorbar(ticks= np.linspace(0.4,1,7))
# plt.title('Noise Correlation Matrix', fontsize=12)
# plt.clim(vmin=0.4, vmax=1) 

plt.show()