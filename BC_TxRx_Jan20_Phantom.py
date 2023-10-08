#Phantom image acquired Jan 20
#Coil: Birdcage transmit/receive

import suspect
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import spect_functions as sfun
import pandas as pd
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import pickle

first_order_phase=-1.4
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


# Main
if __name__=='__main__':
    #Import data

    # #16-Leg BC 31P phantom data (acquired Jan 20)
    data= suspect.io.load_twix('/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31P_BC_TxRx/Phantom/Jan20/meas_MID00041_FID13728_csi_fid.dat') #Shape (16,2,8,16,2048)

    #Step 1: Create noise matrix for data whitening 16-leg data
    noise=data[:,:,:,:,-200:] #take last 200 fid points from all FIDs of first average
    noise= np.moveaxis(noise,4,0).reshape(1, -1) #create noise matrix where each row is a channel 
    noise= noise[:, noise[0]!=0] #remove empty parts of k-space

    #Step 2: Average Acquisitions
    data= np.mean(data, axis=1)

    #Step 3: Perform spatial FT
    spatial_FT= np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data, axes=(0,1,2)), axes=(0,1,2)), axes=(0,1,2))
    spatial_FT= data.inherit(spatial_FT)

    #Apodize FIDS for display purposes
    #Apodize FIDs with exponential filter, for visual diplay purposes only
    spatial_FT_filtered= spatial_FT[:,:,:]*sfun.apod(data,40)

    #Step 4: Frequency, Phase Adjustments, and divide by SD of noise
    #Zero-Order Phase Adjustment 
    for i in range(16):
        for j in range(8):
            for k in range(16):
                spatial_FT[i,j,k]=spatial_FT[i,j,k].adjust_phase(-first_order_phase)
                spatial_FT_filtered[i,j,k]= spatial_FT_filtered[i,j,k].adjust_phase(-first_order_phase)

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

                #divide by sd of noise, such that area under peak directly gives SNR
                spatial_FT[i,j,k] /= np.std(spatial_FT[i,j,k][-200:])

    #### Step 4: FT along FIDs
    spectrum= np.fft.fftshift(np.fft.fft(spatial_FT, axis=-1), axes=-1)
    spectrum= spatial_FT.inherit(spectrum)

    spectrum_filtered= np.fft.fftshift(np.fft.fft(spatial_FT, axis= -1), axes= -1)
    spectrum_filtered= spatial_FT.inherit(spectrum_filtered)

    ### Step 5: Create SNR Map
    SNR_map= np.zeros((16,8,16),'complex')
    for i in range(16):
        for j in range(8):
            for k in range(16):
                SNR_map[i,j,k]= np.sum(np.real(spectrum_filtered[i,j,k][995:1050]))

    #SNR_map= np.abs(SNR_map[:,:,:]/np.amax(SNR_map))
    SNR_map= np.abs(SNR_map[:,:,:])/107637.379 #divide by amax(snr_map) of bc/24-ch receive phantom acquisition


    # #Plot SNR_map along axial slice
    fig4= plt.figure()
    # col_norm= SqueezedNorm(0, 1, 0.3, 1, 1)
    colors= ["black", "red", "orange", "yellow"]
    cmap= LinearSegmentedColormap.from_list('Custom', colors, N=512)
    # snr_plot2=plt.imshow(SNR_map[:,4,:], norm=col_norm, aspect="auto")
    snr_plot2=plt.imshow(SNR_map[:,4,:], cmap=cmap, vmin=0, vmax=1)
    cbar4= plt.colorbar(snr_plot2)
    cbar4.set_label('SNR (A.U.)')
    plt.title('SNR Map: Axial Slice (BC Tx and Rx)', weight= 'bold')

    with open('SNR_map_BCTXRX.pkl', 'wb') as fp:
        pickle.dump(SNR_map, fp)
        print('SNR map dumped to file')
    
    # #Plot spectroscopic image along axial slice
    # fig5, ax= plt.subplots(16,16, figsize=(12,12), sharey= True)
    # fig5.subplots_adjust(hspace=0, wspace=0)
    # fig5.suptitle('31P CSI Image: Axial Slice', weight='bold')
    # for i in range(16):
    #     for j in range(16):
    #         ax[i,j].plot(np.real(spectrum_filtered[i,4,j]))
    #         ax[i,j].set_axis_off()

    ##Plot single spectrum
    #fig6= plt.figure()
    #plt.plot(np.real(spectrum_filtered[8,3,8]))
    #plt.xlim([15,-15])
    #plt.xlabel('ppm', fontsize= 12)
    #plt.ylabel('Intensity (A.U.)', fontsize=12)

    plt.show()




