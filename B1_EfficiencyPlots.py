import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Import 16-rung HF B1+ efficiency plot
B1_eff_16Rung= pd.read_excel("/Users/johnnyderhovagimian/Desktop/B1_Efficiency_Plots/16Rung_B1_Eff_Head_Foot.xlsx")

col1_title= B1_eff_16Rung.columns[0]
col2_title= B1_eff_16Rung.columns[1]
y_pos= B1_eff_16Rung[col1_title]-140 #position
B1_eff= B1_eff_16Rung[col2_title] #B1 Efficiency


#Import 8-rung HF B1+ efficiency plot
B1_eff_8Rung= pd.read_excel("/Users/johnnyderhovagimian/Desktop/B1_Efficiency_Plots/8RungB1_Eff_Head_Foot.xlsx")

y_pos2= B1_eff_8Rung[B1_eff_8Rung.columns[0]]-125
B1_eff2= B1_eff_8Rung[B1_eff_8Rung.columns[1]]


fig1= plt.figure()
plt.plot(y_pos, B1_eff, label= "16-Rung BC Coil")
plt.plot(y_pos2, B1_eff2, label="8-Rung BC Coil")
plt.title('B1+ Efficiency Along HF Direction', weight= 'bold')
plt.xlabel('Distance from Isocentre [mm]')
plt.ylabel('B1+ Efficiency [T/sqrt(W)]')
plt.grid()
plt.legend()

#Impot 8-rung AP B1+ Efficiency Plot
B1_8rung_AP= pd.read_excel("/Users/johnnyderhovagimian/Desktop/B1_Efficiency_Plots/8Rung_B1_Eff_AP.xlsx")

y_pos3= B1_8rung_AP[B1_8rung_AP.columns[0]]
B1_eff3= B1_8rung_AP[B1_8rung_AP.columns[1]]

#Import 16-rung AP B1+ Efficiency Plot
B1_16rung_AP= pd.read_excel("/Users/johnnyderhovagimian/Desktop/B1_Efficiency_Plots/16Rung_B1_Eff_AP.xlsx")

y_pos4= B1_16rung_AP[B1_16rung_AP.columns[0]]
B1_eff4= B1_16rung_AP[B1_16rung_AP.columns[1]]

fig2= plt.figure()
plt.plot(y_pos4, B1_eff4, label= "16-Rung BC Coil")
plt.plot(y_pos3, B1_eff3, label="8-Rung BC Coil")
plt.title("B1+ Efficiency Along AP Direction", weight= 'bold')
plt.xlabel("Y Position [mm]")
plt.ylabel("B1+ Efficiency [uT/\sqrt(W)]")
plt.grid()
plt.legend()













plt.show()



