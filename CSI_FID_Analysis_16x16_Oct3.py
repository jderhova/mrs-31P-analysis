import suspect
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import spect_functions as sfun
import pandas as pd

first_order_phase=0
zero_phase= 0

if __name__=='__main__':
    #Import data

    # #16-Leg BC 31P phantom data (acquired October 3rd)
    data= suspect.io.load_twix('/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31P_Data_24Channel/Oct3/meas_MID00074_FID04501_csi_fid.dat') #Shape (16,2,8,16,24,1024)

    # #8-Leg BC 31P phantom data (acquired December 7th)
    # data_8leg= suspect.io.load_twix('/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31PData_PrototypeCoil/Dec7_Phantom/Dec7/meas_MID00046_FID11300_csi_fid.dat')

    # # data= np.mean(data, axis= 1)
    # #Plot non-zero k-space points-> sanity check, make sure data is elliptically sampled
    # #k-space data from single channel
    # # data_ch2= data[:,1,:,:,1,:]
    # # kspace_sampling=np.zeros((16,8,16)) #create empty array to store 1s where k-space was sampled

    # # for i in range (16):
    # #     for j in range(8):
    # #         for k in range(16):
    # #             if(data_ch2[i,j,k,0]!=0):
    # #                 kspace_sampling[i,j,k]= 1

    # # x,y,z= np.mgrid[-8:8,-4:4,-8:8]
    # # fig= plt.figure()
    # # ax= fig.add_subplot(111,projection= '3d')
    # # img= ax.scatter(x,y,z,c=kspace_sampling, cmap='YlOrRd', alpha=1)
    # # plt.title('K-Space Sampling')

    # ####_____ 8-leg BC 31P CSI data analysis _______####
    # zero_phase_8leg= 3.7

    # # Step 1: take the mean of the two acquisitions
    # data_8leg= np.mean(data_8leg, axis=1) 

    # #Step 2: Perform a spatial fourier transform
    # spatial_FT_8leg= np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data_8leg, axes=(0,1,2)), axes=(0,1,2)), axes=(0,1,2))
    # spatial_FT_8leg= data_8leg.inherit(spatial_FT_8leg)

    # #Step 3: Preprocessing
    # spatial_FT_8leg_filtered= spatial_FT_8leg[:,:,:]*sfun.apod(data_8leg,40) #Apodize data (for display purposes only)

    # #Zero-order phase adjustment
    # for i in range(16):
    #     for j in range(8):
    #         for k in range(16):
    #             spatial_FT_8leg[i,j,k]= spatial_FT_8leg[i,j,k].adjust_phase(-zero_phase_8leg)
    #             spatial_FT_8leg_filtered[i,j,k]= spatial_FT_8leg_filtered[i,j,k].adjust_phase(-zero_phase_8leg)
    
    # #Frequence adjustment to centre peak at 0ppm
    # for i in range(16):
    #     for j in range(8):
    #         for k in range(16):
    #             FID= spatial_FT_8leg[i,j,k]
    #             FID_filtered= spatial_FT_8leg_filtered[i,j,k]
    #             df= suspect.processing.frequency_correction.residual_water_alignment(FID*sfun.apod(data,10))
    #             df_filtered= suspect.processing.frequency_correction.residual_water_alignment(FID_filtered*sfun.apod(data,10))

    #             spatial_FT_8leg[i,j,k]= spatial_FT_8leg[i,j,k].adjust_frequency(-df)
    #             spatial_FT_8leg_filtered[i,j,k]= spatial_FT_8leg_filtered[i,j,k].adjust_frequency(-df_filtered)

    # #FT along FIDs
    # spectrum_8leg= np.fft.fftshift(np.fft.fft(spatial_FT_8leg, axis=-1), axes=-1)
    # spectrum_8leg= spatial_FT_8leg.inherit(spectrum_8leg)

    # spectrum_filtered_8leg= np.fft.fftshift(np.fft.fft(spatial_FT_8leg_filtered, axis=-1), axes=-1)
    # spectrum_filtered_8leg= spatial_FT_8leg.inherit(spectrum_filtered_8leg)

    # #Plot Axial Slice
    # fig1, ax1= plt.subplots(16,16, sharey=True)
    # fig1.subplots_adjust(hspace=0, wspace=0)
    # fig1.suptitle('31P Phantom CSI: Axial Slice (Prototype Coil)', weight='bold')
    # for i in range(16):
    #     for j in range(16):
    #         ax1[i,j].plot(np.real(spectrum_filtered_8leg[i,3,j]))
    #         ax1[i,j].set_axis_off()


    # #Generate SNR Plot
    # snr_map_8leg= np.zeros((16,8,16))
    # for i in range(16):
    #     for j in range(8):
    #         for k in range(16):
    #             snr_map_8leg[i,j,k]= np.sum(np.real(spectrum_8leg[i,j,k][470:574]))/np.std(np.real(spectrum_8leg[i,j,k][-100:]))
            
    #         if (snr_map_8leg[i,j,k] < 0):
    #             snr_map_8leg[i,j,k]= 0.01

    # # snr_map_8leg /= np.max(snr_map_8leg)

    # fig2= plt.figure()
    # plt.plot(data_8leg.frequency_axis_ppm(),np.real(spectrum_filtered_8leg[7,3,7]))
    # plt.xlim([15,-15])
    # plt.xlabel('ppm', fontsize= 12)
    # plt.ylabel('Intensity (A.U.)', fontsize=12)
    

    # #Plot SNR along axial slice
    # fig3= plt.figure()
    # plt.title("SNR Map: Axial Slice", weight= 'bold')
    # snr_map_axial= snr_map_8leg[:,4,:]
    # snr_plot=plt.imshow(snr_map_axial)
    # cbar3= plt.colorbar(snr_plot)   
    # cbar3.set_label('SNR (A.U.)')



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

    #### Step 4: Combine Data using "Weighted by First Point" Method
    cc_wfp= spatial_FT.inherit(np.zeros((16,8,16,data[-1].np), 'complex'))

    for i in range(16):
        for j in range(8):
            for k in range(16):
                cc_wfp[i,j,k]= sfun.phase_and_scale(sfun.combine(spatial_FT[i,j,k], sfun.first(spatial_FT[i,j,k]), first_order_phase, zero_phase))

    cc_wfp= spatial_FT.inherit(cc_wfp)
    cc_wfp_filtered= cc_wfp[:,:,:]*sfun.apod(data,40)

    #### Step 4: FT along FIDs
    spectrum_cc_wfp= np.fft.fftshift(np.fft.fft(cc_wfp, axis=-1), axes=-1)
    spectrum_cc_wfp= cc_wfp.inherit(spectrum_cc_wfp)

    ### Step 5: Create SNR Map
    SNR_map= np.zeros((16,8,16),'complex')
    for i in range(16):
        for j in range(8):
            for k in range(16):
                SNR_map[i,j,k]= np.sum(np.real(spectrum_cc_wfp[i,j,k][472:552]))

    # SNR_map= np.abs(SNR_map[:,:,:]/np.amax(SNR_map))
    SNR_map= np.abs(SNR_map[:,:,:])

    #Plot SNR_map along axial slice
    fig4=plt.figure()
    snr_plot2=plt.imshow(SNR_map[:,4,:])
    cbar4= plt.colorbar(snr_plot2)
    cbar4.set_label('SNR (A.U.)')
    plt.title('SNR Map: Axial Slice', weight= 'bold')
    
    #Plot spectroscopic image along axial slice
    fig5, ax= plt.subplots(16,16, figsize=(12,12), sharey= True)
    fig5.subplots_adjust(hspace=0, wspace=0)
    fig5.suptitle('31P CSI Image: Axial Slice', weight='bold')
    for i in range(16):
        for j in range(16):
            ax[i,j].plot(np.real(cc_wfp_filtered[i,3,j].spectrum()))
            ax[i,j].set_axis_off()

    #Plot single spectrum
    fig6= plt.figure()
    plt.plot(data.frequency_axis_ppm(),np.real(cc_wfp_filtered[7,3,7].spectrum()))
    plt.xlim([15,-15])
    plt.xlabel('ppm', fontsize= 12)
    plt.ylabel('Intensity (A.U.)', fontsize=12)

    plt.show()




