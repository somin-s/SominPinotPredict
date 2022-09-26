from tensorflow import keras
import joblib
import math
import numpy as np
from numpy import log as ln
import pandas as pd


model1 = joblib.load('model1_final.joblib')
model2 = joblib.load('model2_final.joblib')
model3 = joblib.load('model3_final.joblib')
model4 = joblib.load('model4_final.joblib')
# =============================================================================
# Model 1
# =============================================================================
# Inputs: Vineyard	Vintage	Cluster number	Cluster weight	Total Shoot number	Shoot number > 5mm	Shoot number < 5mm	Blind buds	% Leaf in fruit zone	% Vine canopy	Leaf Area per vine	Leaf Area per m	Berrry weight	

# read inputs from user interface and assign values to parameters Vineyard, Vintage, Cluster_number, Cluster_weight, Total_Shoot_number, Shoot_number>5mm,	Shoot_number<5mm, Blind_buds, Leaf_in_fruit_zone, Vine_canopy, Leaf_Area_per_vine, Leaf_Area_per_m,	Berrry_weight
# =============================================================================
# 
# =============================================================================

data_o = pd.read_csv('synthetic1_viti_log_reduced.csv')

# df = pd.DataFrame(data)
# x1 = pd.DataFrame()
x1 = data_o.iloc[0:100,0:4].copy()
x1.iloc[:,0] = ln(x1.iloc[:,0].values/10+1)
x1.iloc[:,1] = ln(x1.iloc[:,1].values/50+1)
x1.iloc[:,2] = ln(x1.iloc[:,2].values/10+1)
x1.iloc[:,3] = ln(x1.iloc[:,3].values/10+5)
from sklearn.metrics import r2_score

# input1st = [[Cluster_number, Cluster_weight, Total_Shoot_number, Shoot_number_gt_5mm, Leaf_in_fruit_zone, Vine_canopy, Leaf_Area_per_vine, Leaf_Area_per_m, Berrry_weight]]

# Outputs: Yield per vine	Yield per m	Yield per m2
# input1st = [[26, 1.305554954, 17, 19, 0.998516868, 1.00040965, 2.182489916, 2.133136421, 1.000693573]]


out = model1.predict(x1)
# # display out[0], out[1], out[2] on UI

# out = model1.predict(input1st)
out1 = pd.DataFrame(out,columns=['out1','out2','out3'])
out1.iloc[:,0] = (np.exp(out1.iloc[:,0])-2)*10
out1.iloc[:,1] = (np.exp(out1.iloc[:,1])-2)*10
out1.iloc[:,2] = (np.exp(out1.iloc[:,2])-2)*10
r2 = r2_score(data_o.iloc[0:100,4:7], out1)
print(r2)
# print(Yield_per_vine)
# print(Yield_per_m)
# print(Yield_per_m2)

# # =============================================================================
# # Model 2
# # =============================================================================
data_o = pd.read_csv('synthetic1_juice.csv')
x2 = data_o.iloc[0:25,0:6].copy()

x2.iloc[:,0] = ln(x2.iloc[:,0]/10+1)
x2.iloc[:,1] = ln(x2.iloc[:,1]/20+20)
x2.iloc[:,2] = ln(x2.iloc[:,2]/10+1)
x2.iloc[:,3] = ln(x2.iloc[:,3]+2)
x2.iloc[:,4] = ln(x2.iloc[:,4]/1000+10)
x2.iloc[:,5] = ln(x2.iloc[:,5]+2)

# #input2nd = [[Vineyard, Vintage, Cluster_number, Cluster_weight, Total_Shoot_number, Shoot_number_gt_5mm, Shoot_number_lt_5mm, Blind_buds, Leaf_in_fruit_zone, Vine_canopy, Leaf_Area_per_vine, Leaf_Area_per_m, Berrry_weight]]


# input2 = [[11, 2, 22, 3.190761048, 13, 12, 1, 5, 1.005120692, 0.974466773, 0.404230132, 2.7146807, 1.064580577]]
out = model2.predict(x2)

# # outputs --> Berry OD280	Berry OD320	Berry OD520	Juice total soluble solids	Juice pH	Juice primary amino acids	Juice malic acid	Juice tartaric acid	Juice calcium	Juice potassium	Juice alanine	Juice arginine	Juice aspartic acid	Juice serine
out2 = pd.DataFrame(out,columns=['out1','out2','out3','out4','out5','out6','out7','out8','out9','out10','out11','out12','out13','out14'])
out2.iloc[:,0] = (np.exp(out2.iloc[:,0])-1)
out2.iloc[:,1] = (np.exp(out2.iloc[:,1])-1)
out2.iloc[:,2] = out2.iloc[:,2]
out2.iloc[:,3] = (np.exp(out2.iloc[:,3]))
out2.iloc[:,4] = np.exp(out2.iloc[:,4])
out2.iloc[:,5] = (np.exp(out2.iloc[:,5])*100)
out2.iloc[:,6] = (np.exp(out2.iloc[:,6])-1)*10
out2.iloc[:,7] = (np.exp(out2.iloc[:,7]))
out2.iloc[:,8] = (np.exp(out2.iloc[:,8]))*50
out2.iloc[:,9] = (np.exp(out2.iloc[:,9]+6))
out2.iloc[:,10] = (np.exp(out2.iloc[:,10])-2)*100
out2.iloc[:,11] = (np.exp(out2.iloc[:,11])-2)*1000
out2.iloc[:,12] = (np.exp(out2.iloc[:,12])-2)*100
out2.iloc[:,13] = (np.exp(out2.iloc[:,13]))
r2 = r2_score(data_o.iloc[0:25,6:20], out2)
print(r2)

# print(out)
# print(Berry_OD280)
# print(Berry_OD320)
# print(Berry_OD520)
# print(Juice_total_soluble_solids)
# print(Juice_pH)
# print(Juice_primary_amino_acids)
# print(Juice_malic_acid)
# print(Juice_tartaric_acid)
# print(Juice_calcium)
# print(Juice_potassium)
# print(Juice_alanine)
# print(Juice_arginine)
# print(Juice_aspartic_acid)
# print(Juice_serine)

out2_o = out2.copy()
# # =============================================================================
# # Model 3
# # =============================================================================

x3 = out2

x3.iloc[:,0] = ln(out2.iloc[:,0]/10+1)
x3.iloc[:,1] = ln(out2.iloc[:,1]+1)
x3.iloc[:,2] = ln(out2.iloc[:,2]+2)
x3.iloc[:,3] = ln(out2.iloc[:,3]/10)
x3.iloc[:,4] = ln(out2.iloc[:,4])
x3.iloc[:,5] = ln(out2.iloc[:,5]/100)
x3.iloc[:,6] = ln(out2.iloc[:,6]+1)
x3.iloc[:,7] = ln(out2.iloc[:,7])
x3.iloc[:,8] = ln(out2.iloc[:,8]/100+1)
x3.iloc[:,9] = ln(out2.iloc[:,9]/1000+1)
x3.iloc[:,10] = ln(out2.iloc[:,10]/1000+1)
x3.iloc[:,11] = ln(out2.iloc[:,11]/1000+2)
x3.iloc[:,12] = ln(out2.iloc[:,12]/100+3)
x3.iloc[:,13] = ln(out2.iloc[:,13]/200+2)

# # input3rd = [[Berry_OD280, Berry_OD320, Berry_OD520, Juice_total_soluble_solids, Juice_pH, Juice_primary_amino_acids, 
# #            Juice_malic_acid, Juice_tartaric_acid, Juice_calcium, Juice_potassium, Juice_alanine, Juice_arginine, 
# #            Juice_aspartic_acid, Juice_serine]]

out = model3.predict(x3)
# # #outputs --> Wine alcohol	Wine pH	Wine monomeric anthocyanins	Wine total anthocyanin	Wine total phenolics	Polymeric Anthocyanins
# out3 = pd.DataFrame(out,columns=['out1','out2','out3','out4','out5','out6'])

out3 = pd.DataFrame(out,columns=['out1','out2','out3','out4','out5'])
out3.iloc[:,0] = (np.exp(out3.iloc[:,0]))
out3.iloc[:,1] = (np.exp(out3.iloc[:,1]))
out3.iloc[:,2] = np.exp(out3.iloc[:,2]*4)
out3.iloc[:,3] = (np.exp(out3.iloc[:,3])-1)*500
out3.iloc[:,4] = (np.exp(out3.iloc[:,4])-1)*20

out03 = out3.copy()
# # r2 = r2_score(data_o.iloc[0:10,13:27], out2)
# # print(r2)


# # print(Wine_alcohol)
# # print(Wine_pH)
# # print(Wine_monomeric_anthocyanins)
# # print(Wine_total_anthocyanin)
# # print(Wine_total_phenolics)
# # print(Polymeric_Anthocyanins)

# # # =============================================================================
# # Model 4
# # =============================================================================
# input4th = [[Wine_alcohol, Wine_pH, Wine_monomeric_anthocyanins, Wine_total_anthocyanin, Wine_total_phenolics, Polymeric_Anthocyanins]]
x4 = out3

x4.iloc[:,0] = ln(out3.iloc[:,0]/10)
x4.iloc[:,1] = ln(out3.iloc[:,1])
x4.iloc[:,2] = ln(out3.iloc[:,2]/100)
x4.iloc[:,3] = ln(out3.iloc[:,3]/100)
x4.iloc[:,4] = ln(out3.iloc[:,4]/10)

out = model4.predict(out3)
out4 = pd.DataFrame(out,columns=['out1'])
out = np.exp(out4.iloc[:,0])+1
# # output --> Quality

# Quality = out
print(out)
