import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from time import sleep

#import data
B_p = pd.read_csv('C:/Users/dongn/DongFile/dong;s junior (WSU)/Second Year - 2023/Semester 2/Machine Learning/Assignment/Assignment 1 - due/bloodpressure-23.csv')


#Variables 
X = B_p[['AGE','ED-LEVEL','SMOKING STATUS', 'EXERCISE', 'WEIGHT', 'SERUM-CHOL', 'IQ', 'SODIUM','GENDER','MARITAL-STATUS']]
X['GENDER'].replace(['M','F'],[1,2], inplace = True)
X['MARITAL-STATUS'].replace(['D','M','S','W'],[1,2,3,4], inplace = True)
X = X.values
y = B_p[['SYSTOLIC']].values

#Ridge model
Rig_model = Ridge(alpha = 0.1)
Rig_model = Rig_model.fit(X,y)
Rig_cv = cross_val_score(Rig_model, X, y, scoring = 'neg_mean_squared_error', cv = 10)
Rig_rmse_scores = np.sqrt(-Rig_cv)
mean_Rig_rmse_scores = Rig_rmse_scores.mean()



#-------------------Streamlit----------------#
st.write("# Welcome patient number #1 of this website")
st.write("My name is Quang, and this website is dedicated to predicting your systolic blood pressure using Machine Learning model (Ridge Regression)")
patient_name = st.text_input("Please enter your name")
patient_age = int(st.number_input("Please enter your age (*number only)", min_value = 0, max_value = 200))
patient_gender = int(st.number_input("What is your biological sex ? (1 for Male or 2 for Female)", min_value = 1, max_value = 2))
patient_marital = int(st.number_input("Are you married ? (1 for Divorced, 2 for Married, 3 for Single, 4 for Widowed)", min_value = 1, max_value = 4))
patient_edudcation = int(st.number_input("Please specify your education level (0 for no education, 1, for primary, 2 for secondary, 3 tertiary)", min_value = 0, max_value = 3))
patient_sm_stus = int(st.number_input("What is your smoking status ? (0 for non-smoker, 1 for light-smoker, 2 for heavy smoker)", min_value = 0, max_value = 2))
patient_exercise = int(st.number_input("Do you exercise ? (0 for no and 1 for yes)", min_value = 0, max_value = 1))
patient_weight = st.number_input("Please enter your weight", min_value = 1)


st.write("For the next section, before attempting to answer, please consult with your doctor first to discuss about your health status")
patient_IQ = int(st.number_input("IQ level", min_value = 0, max_value = 300))
patient_serumchol = int(st.number_input("Serum cholesterone level", min_value = 0))
patient_sodium = int(st.number_input("Sodium level", min_value = 0))

analyse_button = st.button("Show my result")


if analyse_button == True:
    a = np.array([[patient_age, patient_edudcation, patient_sm_stus, patient_exercise, patient_weight, patient_serumchol, patient_IQ, patient_sodium, patient_gender,patient_marital]])
    a = pd.DataFrame(a)
    st.write(f"Your Systolic level is: {Rig_model.predict(a)[0][0]}")


