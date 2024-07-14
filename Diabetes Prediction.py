# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 12:54:11 2024

@author: KIIT0001
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("C:/Users/KIIT0001/Downloads/diabetes_model.sav", 'rb'))

def prediction(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_predict = loaded_model.predict(input_data)
    print("Prediction for the given input:", input_predict)

    if input_predict == 0:
        return "No Diabetes"
    else:
        return "Diabetes detected"

def main(): 
    st.title('Diabetes Prediction')
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age')
    
    diagnosis = ''
    if st.button('Test Result:'):
        diagnosis = prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()

