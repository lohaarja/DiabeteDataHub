# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:53:27 2024

@author: KIIT0001
"""

import numpy as np
import pickle

loaded_model = pickle.load(open("C:/Users/KIIT0001/Downloads/diabetes_model.sav", 'rb'))

input_data = np.array([4, 110, 92, 0, 0, 37.6, 0.191, 30]).reshape(1, -1)
input_predict = loaded_model.predict(input_data)
print("Prediction for the given input:", input_predict)

if input_predict == 0:
    print("No Diabetes")
else:
    print("Diabetes")
