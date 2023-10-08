#Phantom CSI acuisition (16x16)
#Acquired Feb 24
#Coil: 16-rung BC/24-channel Rx
#To be compared to BC-only phantom CSI acquisition

import suspect
import numpy as np
import pickle
import scipy.signal
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import spect_functions as sfun
import pandas as pd
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

first_order_phase=0
zero_phase= 0

#Class to customize colourbar colour gradient
class SqueezedNorm(colors.Normalize):
    def __init__(self, _vmin=None, _vmax=None, _mid=0, _s1=2, _s2=2, _clip=False):
        self._vmin = _vmin # minimum value
        self._mid  = _mid  # middle value
        self._vmax = _vmax # maximum value
        self._s1=_s1; self._s2=_s2
        f = lambda x, zero,_vmax,s: np.abs((x-zero)/(_vmax-zero))**(1./s)*0.5
        self.g = lambda x, zero,_vmin,_vmax, _s1,_s2: f(x,zero,_vmax,_s1)*(x>=zero) - \
                                             f(x,zero,_vmin,_s2)*(x<zero)+0.5
        colors.Normalize.__init__(self, _vmin, _vmax, _clip)

    def __call__(self, value, _clip=None):
        r = self.g(value, self._mid,self._vmin,self._vmax, self._s1,self._s2)
        return np.ma.masked_array(r)


if __name__=='__main__':
    #Import data

    # #16-Leg BC 31P phantom data (acquired October 3rd)
    data= suspect.io.load_twix('/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31P_Data_24Channel/Johnny_Feb24/meas_MID00034_FID16552_csi_fid_16x16.dat') #Shape (16,2,8,16,24,1024)

######______ 16-Leg BC 31P CSI data analysis ______########
    #### Step 1: Create noise matrix for data whitening 16-leg data
    noise=data[:,:,:,:,:,-200:] #take last 200 fid points from all FIDs of first average
    noise= np.moveaxis(noise,4,0).reshape(24, -1) #create noise matrix where each row is a channel 
    noise= noise[:, noise[0]!=0] #remove empty parts of k-space


    #### Step 2: Whiten Data and average acquisitions
    data= np.mean(data, axis=1)
    whitened_data= suspect.processing.channel_combination.whiten(data, noise)
    # whitened_data= np.swapaxes(whitened_data,0,2)

    #### Step 3: Perform spatial FT
    spatial_FT= np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(whitened_data, axes=(0,1,2)), axes=(0,1,2)), axes=(0,1,2))
    spatial_FT= data.inherit(spatial_FT)

    # for i in range(24):
    #     plt.figure()
    #     plt.plot(np.real(data[7,4,7,i]))

    # Step 4: Combine Data using "Weighted by First Point" Method
    cc_wfp= spatial_FT.inherit(np.zeros((16,8,16,data[-1].np), 'complex'))

    for i in range(16):
        for j in range(8):
            for k in range(16):
                cc_wfp[i,j,k]= sfun.phase_and_scale(sfun.combine(spatial_FT[i,j,k], sfun.first(spatial_FT[i,j,k])), zero_phase, first_order_phase)

    cc_wfp= spatial_FT.inherit(cc_wfp)
    cc_wfp_filtered= cc_wfp[:,:,:]*sfun.apod(data,40)

    # Step 4: FT along FIDs
    spectrum_cc_wfp= np.fft.fftshift(np.fft.fft(cc_wfp, axis=-1), axes=-1)
    spectrum_cc_wfp= cc_wfp.inherit(spectrum_cc_wfp)

    # Step 5: Create SNR Map
    SNR_map= np.zeros((16,8,16),'complex')
    for i in range(16):
        for j in range(8):
            for k in range(16):
                SNR_map[i,j,k]= np.sum(np.real(spectrum_cc_wfp[i,j,k][1000:1048]))

    SNR_map= np.abs(SNR_map[:,:,:]/np.amax(SNR_map))
    #SNR_map= np.abs(SNR_map[:,:,:])

    SNR_map= np.roll(SNR_map, 1, axis=0)
    SNR_map= np.roll(SNR_map, -1, axis=2)
    
    
    # #Plot SNR_map along axial slice
    fig4=plt.figure()
    colors= ["black", "red", "orange", "yellow"]
    cmap= LinearSegmentedColormap.from_list('Custom', colors, N=512)
    snr_plot2=plt.imshow(SNR_map[:,4,:], cmap=cmap, vmin=0, vmax=1)
    cbar4= plt.colorbar(snr_plot2)
    cbar4.set_label('SNR (A.U.)')
    plt.title('SNR Map: Axial Slice (BC Tx/24-channel Rx)', weight= 'bold')


    with open("SNR_map_BCTXRX.pkl", 'rb') as fp:
        SNR_map_BC= pickle.load(fp)

    SNR_profile_BC= SNR_map_BC[7,4,:]
    SNR_profile_24Rx= SNR_map[7,4,:]
    fig5= plt.figure()
    plt.plot(SNR_profile_24Rx, label="BC Tx/24Ch Rx")
    plt.plot(SNR_profile_BC, label="BC Tx/Rx")
    plt.ylabel('SNR [A.U.]')
    plt.legend()
    

    
    # #Plot spectroscopic image along axial slice
    # fig5, ax= plt.subplots(16,16, figsize=(12,12), sharey= True)
    # fig5.subplots_adjust(hspace=0, wspace=0)
    # fig5.suptitle('31P CSI Image: Axial Slice', weight='bold')
    # for i in range(16):
    #     for j in range(16):
    #         ax[i,j].plot(np.real(cc_wfp_filtered[i,3,j].spectrum()))
    #         ax[i,j].set_axis_off()

    #Plot single spectrum
    # fig6= plt.figure()
    # plt.plot(np.real(cc_wfp_filtered[7,3,7].spectrum()))
    # # plt.xlim([15,-15])
    # plt.xlabel('ppm', fontsize= 12)
    # plt.ylabel('Intensity (A.U.)', fontsize=12)

    plt.show()




