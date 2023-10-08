#Analyzing non-selective FID
import suspect
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import spect_functions as sfun
import pandas as pd

#####__________FUNCTION DEFINITIONS__________#####
def phase_correct(fid): #Performs a phase correction by subtracting the phase of the first point
    return fid * np.exp(-1j * np.angle(fid[0]))


def phase_correct_scale(fid): #Performs phase correction as above, and scaled by magnitude of first point
    return fid * np.exp(-1j*np.angle(fid[0]))*np.abs(fid[0])

def window_data(fid, window):
    return fid*window


if __name__== "__main__":    
    #Load data
    data= suspect.io.load_twix('/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31P_Data_24Channel/Feb_1/FID/meas_MID00034_FID14939_fid.dat')
    
    #Average across all repetitions
    averaged_data= data.mean(axis=0)

    window= scipy.signal.tukey(data.np*2, 1)[data.np:] #Apodization

    averaged_data_raw= averaged_data

    averaged_data= np.apply_along_axis(window_data, 1, averaged_data, window) #Apply window to each channel's FID

    for i in range(24):
        plt.figure()
        plt.plot(np.real(averaged_data[i]))

    #### WINDOWED AND raw DATA ####
    # plt.plot(averaged_data[5].real, 'r')
    # plt.plot(averaged_data_windowed[5].real, 'k')
    # plt.show()

    phase_corrected_data = np.apply_along_axis(phase_correct, 1, averaged_data) #Apply phase correction to each channel
    
    # copy the MRS parameters from averaged_data to phase_corrected_data
    phae_corrected_data= averaged_data.inherit(phase_corrected_data) #Use inherit function to coppy phase corrected data to MRSClass object

    #Phase correct and scale
    phase_corrected_scaled_data = np.apply_along_axis(phase_correct_scale, 1, averaged_data) #Phase correction and scaling

    scaled_data= averaged_data.inherit(phase_corrected_scaled_data)

    #normalize scaled data by mean scaling factor
    scaled_data /= np.mean(np.abs(averaged_data[:,0]))

    channel_combined= scaled_data.mean(axis=0) #combine 24 scaled and phase corrected channels
    unscaled_channel_combination= phase_corrected_data.mean(axis=0) # combine 24 phase corrected channels

    #SVD channel combination
    svd_weights= suspect.processing.channel_combination.svd_weighting(averaged_data) #calculate svd weights
    svd_combined_data= suspect.processing.channel_combination.combine_channels(averaged_data, svd_weights) #combine channels with SVD weights
    
    plt.rc('xtick', labelsize=6)
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(3,1,1)
    plt.title('Individual Channels', fontsize=12)
   
    for i in range(24):
        plt.plot(averaged_data.time_axis(), averaged_data[i].real)
        
    
    plt.subplot(3,1,2)
    plt.title('SVD Combined FID', fontsize=12)
    plt.plot(svd_combined_data.time_axis(), svd_combined_data.real, 'r')

    plt.subplot(3,1,3)
    plt.title('SVD Combined Spectrum', fontsize=12)
    plt.plot(svd_combined_data.frequency_axis(), svd_combined_data.spectrum().real, 'k')

    fig1= plt.figure()
    plt.title('FIDs Weighted by First Point Method')
    for i in range(24):
        plt.plot(scaled_data[i])

    plt.show()

    # plt.figure('SVD Combined Data')
    # plt.plot(svd_combined_data.frequency_axis(), svd_combined_data.spectrum().real, 'r')
    # plt.plot(channel_combined.frequency_axis(), channel_combined.spectrum().real, 'b')

    # plt.show()


    # plt.figure(1)
    # plt.title('Single Channel FID')
    # plt.plot(averaged_data_raw[2].real)
   
    
    
    # plt.figure(2)
    # plt.title('Scaled and Combined Spectrum')
    # plt.xlabel('Frequency (ppm)')
    # plt.ylabel('Amplitude (A.U)')
    # plt.plot(channel_combined.frequency_axis_ppm(), channel_combined.spectrum().real, 'r')
    # # plt.plot(unscaled_channel_combination.frequency_axis(), unscaled_channel_combination.spectrum().real, 'k')
    # #plt.xlim([10,-1])
    plt.show()
