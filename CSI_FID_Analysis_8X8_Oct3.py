from turtle import end_fill, title
import suspect
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import spect_functions as sfun

global_first_phase= 2*np.pi*0.35e-3
zero_phase=0

if __name__=='__main__':

    data= suspect.io.load_twix('/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31P_Data_24Channel/Oct3/meas_MID00073_FID04500_csi_fid.dat') #data organized by (row, slice, col, ch, fids)


    #Plot non-zero k-space points-> sanity check, make sure data is elliptically sampled
    #k-space data from single channel
    data_ch2= data[:,:,:,1,:]
    kspace_sampling=np.zeros((8,8,8,1)) #create empty array to store 1s where k-space was sampled

    for i in range(8):
        for j in range(8):
            for k in range(8):
                if(data_ch2[i,j,k,0]!=0):
                    kspace_sampling[i,j,k,0]=1
               
    x,y,z= np.mgrid[-4:4,-4:4,-4:4]

    fig= plt.figure()
    ax= fig.add_subplot(111,projection= '3d')
    img= ax.scatter(x,y,z,c=kspace_sampling, cmap='YlOrRd', alpha=1)
    plt.title('K-Space Sampling')

    #### Step 1: Whiten Data
    noise_data= data[:,:,:,:, -100:]
    noise_data= np.moveaxis(noise_data,3,0).reshape(24,-1) #create matrix where rows are channels & columns are noise
    noise_data= noise_data[:, noise_data[0]!=0] #remove unsampled k-space points

    white_data= suspect.processing.channel_combination.whiten(data, noise_data)

    #### STEP 2: Apply FT along spatial dimensions, Siemens data ordering and numpy data ordering differ, "massaging" data based on Rowland's code
    spatial_FT= np.fft.ifftn(np.fft.fftshift(np.swapaxes(white_data,0,2),axes=(0,1,2)), axes=(0,1,2))
    spatial_FT= np.fft.fftshift(spatial_FT, axes=(0,1,2))
    spatial_FT= data.inherit(spatial_FT)
    
    ### STEP 3: Channel combination by first point method, each voxel will be normalized by its combined noise
    cc_wfp= spatial_FT.inherit(np.zeros((8,8,8,data[-1].np),'complex')) #matrix to store combined results

    for i in range(8):
        for j in range(8):
            for k in range(8):
                cc_wfp[i,j,k]= sfun.phase_and_scale(sfun.combine(spatial_FT[i,j,k], sfun.first(spatial_FT[i,j,k]), global_first_phase,zero_phase))

    cc_wfp= data.inherit(cc_wfp)
    cc_wfp_filtered= cc_wfp[:,:,:]*sfun.apod(data,40)
   
    # Plot combined fid in axial slice
    fig2= plt.figure()
    fig2.subplots(8,8, sharey= True)
    count=1
    for i in range(8):
        for j in range(8):
            plt.subplot(8,8,count)
            plt.plot(np.abs(cc_wfp[i,4,j,:]))
            count += 1
    

    # #FT along FID
    spectrum_cc_wfp= np.fft.fftshift(np.fft.fft(cc_wfp, axis=-1), axes=-1)
    spectrum_cc_wfp= cc_wfp.inherit(spectrum_cc_wfp)

    #Create SNR map
    SNR_map= cc_wfp.inherit(np.sum(np.abs((cc_wfp[:,:,:]*sfun.apod(data, 40)).spectrum())[:,:,:,472:552], axis=-1)).astype(np.int32)

    SNR_map= SNR_map[:,:,:]/np.amax(SNR_map)

    fig3=plt.figure()
    snr_plot=plt.imshow(SNR_map[1:,1:,4], vmax=1, vmin=0)
    cbar3= plt.colorbar(snr_plot)
    cbar3.set_label('SNR A.U.')
    plt.title('SNR Map Axial Slice Through Phantom')
    
    fig4= plt.figure()
    fig4.subplots(8,8, sharey= True)
    count= 1
    for i in range(8):
        for j in range(8):
            plt.subplot(8,8,count)
            plt.plot(cc_wfp_filtered[i,4,j].spectrum().real)
            count += 1

    # ## Plotting individual channel spectra
    # fig5= plt.figure()
    # fig5.subplots(8,8, sharey= True)
    # count= 1
    # for i in range(8):
    #     for j in range(8):
    #         plt.subplot(8,8,count)
    #         plt.plot(np.abs((spatial_FT[i,j,4,20,:]*sfun.apod(data,40)).spectrum()))
    #         count += 1


    plt.show()

