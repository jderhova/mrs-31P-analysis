from turtle import end_fill, title
import suspect
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import spect_functions as sfun
#Reconstructing GRE image

# data= suspect.io.load_twix("/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31P_Data_24Channel/Johnny_Oct25/20221025/meas_MID00068_FID07321_csi_fid.dat")

GRE= suspect.image.load_dicom_volume('/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31P_Data_24Channel/Johnny_Oct25/20221025/31P_OCT25_22_10_25-13_43_26-DST-1_3_12_2_1107_5_2_0_79117/SEQUENCE_DEVELOPMENT_JOHNNY_20221025_134754_841000/T1_MPRAGE_SAG_P2_0_70MM_0004/31P_OCT25.MR.SEQUENCE_DEVELOPMENT_JOHNNY.0004.0001.2022.10.25.15.48.05.974329.127172703.IMA')

# t1= GRE.resample(data.sagittal_vector, data.coronal_vector, (1,256,256), centre= data.to_scanner(0,0,0))


plt.imshow(GRE[112],cmap= plt.cm.gray ) #GRE data stored as slice, row column

plt.show()