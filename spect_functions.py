import numpy as np
import scipy.signal
import suspect

## Useful functions for spectroscopic analysis

def windowData(fid, window): #applied provided window to the FID data 
    return fid*window

def phase_correct_scale(fid): #Performs phase correction as above, and scaled by magnitude of first point
    return fid * np.exp(-1j*np.angle(fid[0]))*np.abs(fid[0])

def apod(FID_data, scale):
    return np.exp(-FID_data.time_axis()*scale)

def combine (sv, weights): #May need to add adjust phase here for global phase shift based on TE
    return suspect.processing.channel_combination.combine_channels(sv, weights) #.adjust_phase(zero_phase, global_phase)

def phase_and_scale(sv, zero_phase, global_phase): #input is combined FID for single voxel
    #noise estimation
    sv_noise= np.std(sv[-100:]) 
    #phase correction parameters
    # zp,fp= suspect.processing.phase.mag_real(sv*apod(sv,10), range_hz=(-2000,2000))
    #frequency correction placing highest peak at 0Hz
    df= suspect.processing.frequency_correction.residual_water_alignment(sv*apod(sv,10))

    return sv.adjust_frequency(-df).adjust_phase(zero_phase, global_phase)/sv_noise 

def first(sv):
    return sv[:,0].conj()



