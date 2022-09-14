# Importing required libraries, obviously
import streamlit as st
import pandas as pd 
import joblib
import numpy as np
import altair as alt

#import cv2
#from PIL import Image
#import math as mt
#from tensorflow import keras

# Load  model a 
#model_s = keras.models.load_model('model1')
model_1 = joblib.load(open("model1_final.joblib","rb"))
model_2 = joblib.load('model2_final.joblib')
model_3 = joblib.load('model3_final.joblib')
model_4 = joblib.load('model4_final.joblib')


st.write("""
# Wine Quality Prediction
This app predicts the ** Quality of Wine **  using **wine features** input via the **select boxes and side panels** 
""")
st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar

def get_user_input_forPredict(): #increased by 1; Somin 01092022

    Vineyard = st.sidebar.selectbox("Select Vinyard",("Marlborough_A", 
                                                        "Marlborough_B",
                                                        "Marlborough_C",
                                                        "Marlborough_D",
                                                        "Otago_A",
                                                        "Otago_B",
                                                        "Otago_C",
                                                        "Otago_D",
                                                        "Wairarapa_A",
                                                        "Wairarapa_B",
                                                        "Wairarapa_C",
                                                        "Wairarapa_D"
                                                        )) 
    Vintage = st.sidebar.selectbox("Select Vintage",("2018", "2019", "2020", "2021"))
    Cluster_number = st.sidebar.slider('Cluster number', 1.0, 52.0, 6.0, 1.0) 
    Cluster_weight = st.sidebar.slider('Cluster weight', 35.0, 253.0, 38.0, 0.1) 
    Total_Shoot_number = st.sidebar.slider('Total Shoot number', 5.0, 35.0, 5.0, 1.0) 
    Shoot_number_more_5mm = st.sidebar.slider('Shoot number> 5mm', 4.0, 30.0, 4.0, 1.0) 
    Shoot_number_less_5mm = st.sidebar.slider('Shoot number<5mm', 1.0, 20.0, 1.0, 1.0) 
    Blind_buds = st.sidebar.slider('Blind buds', 0.0, 15.0, 0.0, 1.0) 
    Leaf_in_fruit_zone = st.sidebar.slider('Leaf in fruit zone', 0.1, 1.0, 0.1, 0.001) 
    Vine_canopy = st.sidebar.slider('Vine_canopy', 0.3, 1.0, 1.0, 0.001) 
    Leaf_Area_per_vine = st.sidebar.slider('Leaf Area/vine', 3240.0, 52000.0, 320.0, 1.0) 
    Leaf_Area_per_m = st.sidebar.slider('Leaf Area/m', 2800.0, 32000.0, 2800.0, 1.0) 
    Berrry_weight = st.sidebar.slider('Berrry weight', 1.0, 25.0, 1.0, 0.001) 
    
    
    features = {'Vineyard': Vineyard,
            'Vintage': Vintage,
            'Cluster_number': Cluster_number,
            'Cluster_weight': Cluster_weight,
            'Total_Shoot_number': Total_Shoot_number,
            'Shoot_number_more_5mm': Shoot_number_more_5mm,
            'Shoot_number_less_5mm': Shoot_number_less_5mm,
            'Blind_buds': Blind_buds,
            'Leaf_in_fruit_zone': Leaf_in_fruit_zone,
            'Vine_canopy': Vine_canopy,
            'Leaf_Area_per_vine': Leaf_Area_per_vine,
            'Leaf_Area_per_m': Leaf_Area_per_m,
            'Berrry_weight': Berrry_weight
            }
    data = pd.DataFrame(features,index=[0])

    return data


def OutputFirstModule(prediction_proba):
    Yield_per_vine = np.exp(prediction_proba[0]-2)*10
    Yield_per_m = np.exp(prediction_proba[1]-2)*10
    Yield_per_m2 = np.exp(prediction_proba[2]-2)*10

    source = pd.DataFrame({
        'Value': ['Yield/vine', 'Yield/m', 'Yield/m2'],
        'Rate': [Yield_per_vine, Yield_per_m, Yield_per_m2]
     })
 
    bar_chart = alt.Chart(source, width=450, height=200).mark_bar(clip=True).encode(x='Rate', y='Value')

    st.table(source)#show table
    st.altair_chart(bar_chart)

    return

def OutputSecond_Third_FourthModule(prediction_proba):
    #module2====================================================================================================================
    Berry_OD280 = np.exp(prediction_proba[0])-1
    Berry_OD320 = np.exp(prediction_proba[1])-1
    Berry_OD520 = prediction_proba[2]
    Juice_total_soluble_solids = np.exp(prediction_proba[3])
    Juice_pH = np.exp(prediction_proba[4])
    Juice_primary_amino_acids = np.exp(prediction_proba[5])*100
    Juice_malic_acid = np.exp(prediction_proba[6]-1)*10
    Juice_tartaric_acid = np.exp(prediction_proba[7])
    Juice_calcium = np.exp(prediction_proba[8])*50
    Juice_potassium = np.exp(prediction_proba[9]+6)
    Juice_alanine = np.exp(prediction_proba[10]-2)*100
    Juice_arginine = np.exp(prediction_proba[11]-2)*1000
    Juice_aspartic_acid = np.exp(prediction_proba[12]-2)*100
    Juice_serine = np.exp(prediction_proba[13])

    source2 = pd.DataFrame({
        'Value': ['Berry OD280','Berry OD320','Berry OD520','Juice total soluble solids','Juice pH','Juice primary amino acids','Juice malic acid'
                    ,'Juice tartaric acid','Juice calcium','Juice potassium','Juice alanine','Juice arginine','Juice aspartic acid','Juice serine'],
        'Rate': [Berry_OD280,Berry_OD320,Berry_OD520,Juice_total_soluble_solids,Juice_pH,Juice_primary_amino_acids,Juice_malic_acid
                    ,Juice_tartaric_acid,Juice_calcium,Juice_potassium,Juice_alanine,Juice_arginine,Juice_aspartic_acid,Juice_serine]
     })
    st.table(source2)#show table module2
    #st.write(CheckNegative((mt.exp(prediction_proba[10])-2)*100)) test
    #module3 ==================================================================================================================
    Berry_OD280 = np.log10(np.exp(prediction_proba[0])-1)/10+1
    Berry_OD320 = np.log10(np.exp(prediction_proba[1])-1)+1
    Berry_OD520 = np.log10(prediction_proba[2])+2
    Juice_total_soluble_solids = np.log10(np.exp(prediction_proba[3]))/10
    Juice_pH = np.log10(np.exp(prediction_proba[4]))
    Juice_primary_amino_acids = np.log10(np.exp(prediction_proba[5])*100)/100
    Juice_malic_acid = np.log10((np.exp(prediction_proba[6])-1)*10)+1
    Juice_tartaric_acid = np.log10(np.exp(prediction_proba[7]))
    Juice_calcium = np.log10((np.exp(prediction_proba[8]))*50)/100+1
    Juice_potassium = np.log10(np.exp(prediction_proba[9]+6))/1000+1
    Juice_alanine = np.log10(CheckNegative(((np.exp(prediction_proba[10])-2)*100)))/1000+1
    Juice_arginine = np.log10(CheckNegative((np.exp(prediction_proba[11])-2)*1000))/1000+2
    Juice_aspartic_acid = np.log10(CheckNegative((np.exp(prediction_proba[12])-2)*100))/100+3
    Juice_serine = np.log10(np.exp(prediction_proba[13]))/200+2

    ThirdMol_value = model_3.predict([[Berry_OD280, Berry_OD320, Berry_OD520, Juice_total_soluble_solids, Juice_pH, Juice_primary_amino_acids, 
           Juice_malic_acid, Juice_tartaric_acid, Juice_calcium, Juice_potassium, Juice_alanine, Juice_arginine, 
           Juice_aspartic_acid, Juice_serine]])

    Wine_alcohol = np.exp(ThirdMol_value[0][0])
    Wine_pH = np.exp(ThirdMol_value[0][1])
    Wine_monomeric_anthocyanins = np.exp(ThirdMol_value[0][2]*4)
    Wine_total_anthocyanin = (np.exp(ThirdMol_value[0][3])-1)*500
    Wine_total_phenolics = (np.exp(ThirdMol_value[0][4])-1)*20
    Polymeric_Anthocyanins = (np.exp(ThirdMol_value[0][5])-2)*100

    source3 = pd.DataFrame({
            'Value': ['Wine alcohol', 'Wine pH', 'Wine monomeric anthocyanins', 'Wine total anthocyanin', 'Wine total phenolics', 'Polymeric Anthocyanins'],
            'Rate': [Wine_alcohol,Wine_pH, Wine_monomeric_anthocyanins ,Wine_total_anthocyanin, Wine_total_phenolics, Polymeric_Anthocyanins]
        })
    st.table(source3)#show table module3

    #module4 ==================================================================================================================
    FourthMol_value = model_4.predict([[Wine_alcohol, Wine_pH, Wine_monomeric_anthocyanins, Wine_total_anthocyanin, Wine_total_phenolics, Polymeric_Anthocyanins]])
    source = pd.DataFrame({
        'Quality': [''],
        'Rate': [FourthMol_value[0]]
     })
    
    bar_chart = alt.Chart(source, width=450, height=100).mark_bar(clip=True).encode(x='Rate', y='Quality')
    #bar_chart = alt.Chart(source).mark_bar().encode(
    #    y='Rate',
    #    x='Value',
    #)
    
    #st.table(source)#show table module4
    st.altair_chart(bar_chart)
    st.write(f'Quality rate: ***{round(FourthMol_value[0],2)}***')
    return

def CheckNegative(paraNum):
    if paraNum < 0: tmp = 1
    else: tmp = paraNum
    return tmp

def PreVinyard(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    if df == "Marlborough_A":
        vinyard = 0
    elif df == "Marlborough_B":
        vinyard = 1
    elif df == "Marlborough_C":
        vinyard = 2
    elif df == "Marlborough_D":
        vinyard = 3
    elif df == "Otago_A":
        vinyard = 4
    elif df == "Otago_B":
        vinyard = 5
    elif df == "Otago_C":
        vinyard = 6
    elif df == "Otago_D":
        vinyard = 7
    elif df == "Wairarapa_A":
        vinyard = 8
    elif df == "Wairarapa_B":
        vinyard = 9
    elif df == "Wairarapa_C":
        vinyard = 10
    elif df == "Wairarapa_D":
        vinyard = 11
    return vinyard

def PreVintage(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    if df == "2018":
        vintage = 0
    elif df == "2019":
        vintage = 1
    elif df == "2020":
        vintage = 3
    elif df == "2021":
        vintage = 2
        
    return vintage

Uinp = get_user_input_forPredict()
input1 = PreVinyard(Uinp.get("Vineyard")[0])
input2 = PreVintage(Uinp.get("Vintage")[0])
input3 = pd.to_numeric(Uinp.get("Cluster_number")[0])
input4 = pd.to_numeric(Uinp.get("Cluster_weight")[0])
input5 = pd.to_numeric(Uinp.get("Total_Shoot_number")[0])
input6 = pd.to_numeric(Uinp.get("Shoot_number_more_5mm")[0])
input7 = pd.to_numeric(Uinp.get("Shoot_number_less_5mm")[0])
input8 = pd.to_numeric(Uinp.get("Blind_buds")[0])
input9 = pd.to_numeric(Uinp.get("Leaf_in_fruit_zone")[0])
input10 = pd.to_numeric(Uinp.get("Vine_canopy")[0])
input11 = pd.to_numeric(Uinp.get("Leaf_Area_per_vine")[0])
input12 = pd.to_numeric(Uinp.get("Leaf_Area_per_m")[0])
input13 = pd.to_numeric(Uinp.get("Berrry_weight")[0])

#Preprocess for Module1
Vineyard = input1
Vintage = input2
Cluster_number = input3

Cluster_weight = np.log10(input4/50+1)
Total_Shoot_number = input5
Shoot_number_gt_5mm = input6 
Shoot_number_lt_5mm = input7
Blind_buds = input8
Leaf_in_fruit_zone = np.log10(input9)*(-1)
Vine_canopy = np.log10(input10)*(-1)
Leaf_Area_per_vine = np.log10(input11/10000+1)
Leaf_Area_per_m = np.log10(input12/10000+1)
Berrry_weight = np.log10(input13/10+1)
#Output first module
FirstMol_value = model_1.predict([[Cluster_number, Cluster_weight, Total_Shoot_number, Shoot_number_gt_5mm, Leaf_in_fruit_zone, Vine_canopy, Leaf_Area_per_vine, Leaf_Area_per_m, Berrry_weight]])
OutputFirstModule(FirstMol_value[0])

#Preprocess for Module2
Leaf_in_fruit_zone = np.log10(input9+2)
Vine_canopy = np.log10(input10+2)
Leaf_Area_per_vine = np.log10(input11/10000+1)
Leaf_Area_per_m = np.log10(input12/1000+10)
Berrry_weight = np.log10(input13+2)
#Output second & third module
SecondMol_value = model_2.predict([[Vineyard, Vintage, Cluster_number, Cluster_weight, Total_Shoot_number, Shoot_number_gt_5mm, Shoot_number_lt_5mm, Blind_buds, Leaf_in_fruit_zone, Vine_canopy, Leaf_Area_per_vine, 
                    Leaf_Area_per_m, Berrry_weight]])
OutputSecond_Third_FourthModule(SecondMol_value[0])
