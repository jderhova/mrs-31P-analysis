from turtle import end_fill, title
import suspect
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import spect_functions as sfun
#Reconstructing GRE image

# data= suspect.io.load_twix("/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31P_Data_24Channel/Johnny_Oct25/20221025/meas_MID00068_FID07321_csi_fid.dat")

GRE= suspect.image.load_dicom_volume('/Users/johnnyderhovagimian/Desktop/31PMRS_Data/31P_Data_24Channel/Feb_17/Dicom/SEQUENCE_DEVELOPMENT_JOHNNY_20230217_101628_962000/T1_MP2RAGE_SAG_P3_0_70MM_T1_IMAGES_0007/31P_16CSI_FEB17.MR.SEQUENCE_DEVELOPMENT_JOHNNY.0007.0113.2023.02.17.11.45.31.763841.140285272.IMA')
# t1= GRE.resample(data.sagittal_vector, data.coronal_vector, (1,256,256), centre= data.to_scanner(0,0,0))


plt.imshow(GRE[112],cmap= plt.cm.gray ) #GRE data stored as slice, row column

plt.show()