# Importing required libraries, obviously
import streamlit as st
import pandas as pd 
import joblib
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
#import cv2
#from PIL import Image
#import math as mt
#from tensorflow import keras


model_1 = joblib.load(open("model1_final.joblib","rb"))
model_2 = joblib.load('model2_final.joblib')
model_3 = joblib.load('model3_final.joblib')
model_4 = joblib.load('model4_final.joblib')


st.write("""
# Wine Quality Prediction
This app predicts the ** Quality of Wine **  using **wine features** input via the **side panels** 
""")
st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar

def Modules(prediction_proba1, prediction_proba2):
    #module1====================================================================================================================
    Yield_per_wine = np.exp(prediction_proba1[0]-2)*10
    Yield_per_m = np.exp(prediction_proba1[1]-2)*10
    Yield_per_m2 = np.exp(prediction_proba1[2]-2)*10
    #module2====================================================================================================================
    Berry_OD280 = np.exp(prediction_proba2[0])-1
    Berry_OD320 = np.exp(prediction_proba2[1])-1
    Berry_OD520 = prediction_proba2[2]
    Juice_total_soluble_solids = np.exp(prediction_proba2[3])
    Juice_pH = np.exp(prediction_proba2[4])
    Juice_primary_amino_acids = np.exp(prediction_proba2[5])*100
    Juice_malic_acid = np.exp(prediction_proba2[6]-1)*10
    Juice_tartaric_acid = np.exp(prediction_proba2[7])
    Juice_calcium = np.exp(prediction_proba2[8])*50
    Juice_potassium = np.exp(prediction_proba2[9]+6)
    Juice_alanine = np.exp(prediction_proba2[10]-2)*100
    Juice_arginine = np.exp(prediction_proba2[11]-2)*1000
    Juice_aspartic_acid = np.exp(prediction_proba2[12]-2)*100
    Juice_serine = np.exp(prediction_proba2[13])

    source2 = [Berry_OD280,Berry_OD320,Berry_OD520,Juice_total_soluble_solids,Juice_pH,Juice_primary_amino_acids,Juice_malic_acid
                    ,Juice_tartaric_acid,Juice_calcium,Juice_potassium,Juice_alanine,Juice_arginine,Juice_aspartic_acid,Juice_serine]
    #module3 ==================================================================================================================
    Berry_OD280 = np.log(Berry_OD280/10+1)
    Berry_OD320 = np.log(Berry_OD320+1)
    Berry_OD520 = np.log(Berry_OD520+2)
    Juice_total_soluble_solids = np.log(Juice_total_soluble_solids/10)
    Juice_pH = np.log(Juice_pH)
    Juice_primary_amino_acids = np.log(Juice_primary_amino_acids/100)
    Juice_malic_acid = np.log(Juice_malic_acid+1)

    Juice_tartaric_acid = np.log(Juice_tartaric_acid)
    Juice_calcium = np.log(Juice_calcium/100+1)
    Juice_potassium = np.log(Juice_potassium/1000+1)
    Juice_alanine = np.log(Juice_alanine/1000+1)
    Juice_arginine = np.log(Juice_arginine/1000+2)
    Juice_aspartic_acid = np.log(Juice_aspartic_acid/100+3)
    Juice_serine = np.log(Juice_serine/200+2)

    ThirdMol_value = model_3.predict([[Berry_OD280, Berry_OD320, Berry_OD520, Juice_total_soluble_solids, Juice_pH, Juice_primary_amino_acids, 
           Juice_malic_acid, Juice_tartaric_acid, Juice_calcium, Juice_potassium, Juice_alanine, Juice_arginine, 
           Juice_aspartic_acid, Juice_serine]])

    Wine_alcohol = np.exp(ThirdMol_value[0][0])
    Wine_pH = np.exp(ThirdMol_value[0][1])
    Wine_monomeric_anthocyanins = np.exp(ThirdMol_value[0][2]*4)
    Wine_total_anthocyanin = (np.exp(ThirdMol_value[0][3])-1)*500
    Wine_total_phenolics = (np.exp(ThirdMol_value[0][4])-1)*20


    source3 = [Wine_alcohol,Wine_pH,Wine_monomeric_anthocyanins,Wine_total_anthocyanin,Wine_total_phenolics]
    #module4 ==================================================================================================================
    Wine_alcohol = np.log(Wine_alcohol/10)
    Wine_pH = np.log(Wine_pH)
    Wine_monomeric_anthocyanins = np.log(Wine_monomeric_anthocyanins/100)
    Wine_total_anthocyanin = (np.log(Wine_total_anthocyanin)/100)
    Wine_total_phenolics = (np.log(Wine_total_phenolics)/10)

    FourthMol_value = model_4.predict([[Wine_alcohol, Wine_pH, Wine_monomeric_anthocyanins, Wine_total_anthocyanin, Wine_total_phenolics]])
    quality = np.exp(FourthMol_value[0])+1
    quality = round(quality,2)

    Quality_yieldperwine = [quality, Yield_per_wine, Yield_per_m, Yield_per_m2,source2,source3]
    return Quality_yieldperwine

#=============================================================================================================================== Main
Cluster_number = st.sidebar.slider('Cluster number', 1.0, 52.0, 23.0, 1.0) 
Cluster_weight = st.sidebar.slider('Cluster weight (g)', 35.0, 253.0, 144.0, 1.0) 
Shoot_number_more_5mm = st.sidebar.slider('Shoot number', 4.0, 30.0, 12.0, 1.0) 
Vine_canopy = st.sidebar.slider('Vine canopy (%)', 0.0, 1.0, 0.5, 0.001) 
Leaf_Area_per_m = st.sidebar.slider('Leaf Area/metre', 2800.0, 32000.0, 12000.0, 1.0) 
Berry_weight = st.sidebar.slider('Berry weight (g)', 1.0, 2.0, 1.78, 0.001) 

features = {'Cluster_number': Cluster_number,
            'Cluster_weight': Cluster_weight,
            'Shoot_number_more_5mm': Shoot_number_more_5mm,
            'Vine_canopy': Vine_canopy,
            'Leaf_Area_per_m': Leaf_Area_per_m,
            'Berry_weight': Berry_weight
            }
data = pd.DataFrame(features,index=[0])

Cluster_number_input = pd.to_numeric(data.get("Cluster_number")[0])
Cluster_weight_input = pd.to_numeric(data.get("Cluster_weight")[0])
Shoot_num_input = pd.to_numeric(data.get("Shoot_number_more_5mm")[0])
Vine_canopy_input = pd.to_numeric(data.get("Vine_canopy")[0])
Leaf_area_input = pd.to_numeric(data.get("Leaf_Area_per_m")[0])
Berry_weight_input = pd.to_numeric(data.get("Berry_weight")[0])

Cluster_number_ran = np.random.normal(Cluster_number_input,4.0,20)
Cluster_weight_ran = np.random.normal(Cluster_weight_input,4.0,20)
Shoot_number_more_5mm_ran = np.random.normal(Shoot_num_input,4.0,20)
Vine_canopy_ran = np.random.normal(Vine_canopy_input,0.1,20)
Leaf_Area_per_m_ran = np.random.normal(Leaf_area_input,13.6,20)
Berry_weight_ran = np.random.normal(Berry_weight_input,0.1,20)

Arr_Quality_yield = pd.DataFrame()
Arr_Quality_yield_list = pd.DataFrame()
Arr_model2 = pd.DataFrame()
Arr_model3 = pd.DataFrame()
for i in range(20):

    #Preprocess for Module1
    Cluster_number1 = np.log(Cluster_number_ran[i]/10+1)
    Cluster_weight1 = np.log(Cluster_weight_ran[i]/50+1)
    Shoot_number_gt_5mm1 = np.log(Shoot_number_more_5mm_ran[i]/10+1)
    Berry_weight1 = np.log(Berry_weight_ran[i]/10+5)
    #Output first module
    FirstMol_value = model_1.predict([[Cluster_number1, Cluster_weight1, Shoot_number_gt_5mm1,Berry_weight1]])
    #OutputFirstModule(FirstMol_value[0])

    #Preprocess for Module2
    Cluster_number2 = np.log(Cluster_number_ran[i]/10+1)
    Cluster_weight2 = np.log(Cluster_weight_ran[i]/20+20)
    Shoot_number_gt_5mm2 = np.log(Shoot_number_more_5mm_ran[i]/10+1)
    Vine_canopy2 = np.log(Vine_canopy_ran[i]+2)
    Leaf_Area_per_m2 = np.log(Leaf_Area_per_m_ran[i]/1000+10)
    Berry_weight2 = np.log(Berry_weight_ran[i]+2) 

    #Output second & third module
    SecondMol_value = model_2.predict([[Cluster_number2, Cluster_weight2, Shoot_number_gt_5mm2,Vine_canopy2,Leaf_Area_per_m2,Berry_weight2]])

    cn = round(Cluster_number_ran[i])
    cw = round(Cluster_weight_ran[i])
    sn = round(Shoot_number_more_5mm_ran[i])
    vc = round(Vine_canopy_ran[i],2)
    la = round(Leaf_Area_per_m_ran[i])
    bw = round(Berry_weight_ran[i],2)


    source = Modules(FirstMol_value[0],SecondMol_value[0])
    ##Arr_Quality_yield = Arr_Quality_yield.append({'Quality': source[0],'Yield': "Yield per wine",'Value': source[1], 'Info': "Information", 
    ##'Cluster number':cn,'Cluster weight (g)':cw,'Shoot number':sn,'Vine canopy (%)': vc,'Leaf area / metre':la,'Berry weight (g)':bw},ignore_index=True)
    ##Arr_Quality_yield = Arr_Quality_yield.append({'Quality': source[0],'Yield': "Yield per metre",'Value': source[2], 'Info': "Information", 
    ##'Cluster number':cn,'Cluster weight (g)':cw,'Shoot number':sn,'Vine canopy (%)': vc,'Leaf area / metre':la,'Berry weight (g)':bw},ignore_index=True)
    ##Arr_Quality_yield = Arr_Quality_yield.append({'Quality': source[0],'Yield': "Yield per square metre",'Value': source[3], 'Info': "Information", 
    ##'Cluster number':cn,'Cluster weight (g)':cw,'Shoot number':sn,'Vine canopy (%)': vc,'Leaf area / metre':la,'Berry weight (g)':bw},ignore_index=True)

    quality = round(source[0],2)
    Yield_per_wine = round(source[1],2)
    Yield_per_metre = round(source[2],2)
    Yield_per_metre2 = round(source[3],2)
    Arr_Quality_yield_list = Arr_Quality_yield_list.append({'Quality':quality,'Yield per wine': Yield_per_wine,'Yield per metre': Yield_per_metre,'Yield per square metre': Yield_per_metre2},ignore_index=True)

    ##Arr_model2 = Arr_model2.append({'Berry OD280(AU)':source[4][0],'Berry OD320(AU)':source[4][1],'Berry OD520(AU)':source[4][2],'Juice total soluble solids(oBrix)':source[4][3],'Juice pH':source[4][4],'Juice primary amino acids(g/L)':source[4][5],'Juice malic acid(g/L)':source[4][6],
    ##                'Juice tartaric acid(g/L)':source[4][7],'Juice calcium(mg/L)':source[4][8],'Juice potassium(mg/L)':source[4][9],'Juice alanine(μmol/L)':source[4][10],'Juice arginine(μmol/L)':source[4][11],'Juice aspartic acid(μmol/L)':source[4][12],'Juice serine':source[4][13]},ignore_index=True)
    ##Arr_model3 = Arr_model3.append({'Wine alcohol(% v/v)':source[5][0],'Wine pH':source[5][1],'Wine monomeric anthocyanins(mg/L M3G)':source[5][2],'Wine total anthocyanin(mg/L M3G)':source[5][3],'Wine total phenolics':source[5][4]},ignore_index=True)

#==============================================================================================================================plot graph

#category = px.scatter(Arr_Quality_yield, x="Quality", y="Value", color="Yield", trendline= "ols", hover_name='Info', hover_data=["Cluster number","Cluster weight (g)","Shoot number","Vine canopy (%)","Leaf area / metre","Berry weight (g)"])
category.update_yaxes(range=[2.5,3.5])
category.update_xaxes(range=[4.5,5.5])
st.plotly_chart(category, s=100)

#plot = px.scatter(Arr_Quality_yield, x="Quality",y="Value", color="Yield", facet_col="Yield",hover_name='Info', hover_data=["Cluster number","Cluster weight (g)","Shoot number","Vine canopy (%)","Leaf area / metre","Berry weight (g)"])
plot.update_xaxes(range=[1, 5])
plot.update_yaxes(range=[1, 6])
st.plotly_chart(plot, s=100)

if st.checkbox("Ouput 20 samples"):
    st.table(Arr_model2)
    st.table(Arr_model3)
    

        
