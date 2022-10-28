# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 19:40:11 2022

@author: krist
"""
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from catboost import CatBoostRegressor
import pickle
import numpy as np
st.header("Kiva Loans Prediction App")
st.text_input("Enter your Name: ", key="name")
kiva_loans = pd.read_csv("https://raw.githubusercontent.com/kristophernerl/PredictKivaLoans/main/kiva_loans2.csv")

#load label encoders
label_encoder_sector = LabelEncoder()
label_encoder_sector.classes_ = np.load('label_encoder_sector.npy',allow_pickle=True)
label_encoder_gender = LabelEncoder()
label_encoder_gender.classes_ = np.load('label_encoder_gender.npy',allow_pickle=True)

# load model
cat_model = CatBoostRegressor()
cat_model.load_model("cat_model.json")

##start
if st.checkbox('Show Training Dataframe'):
    kiva_loans

st.subheader("Please select relevant features of the loan you are looking for")
left_column, right_column = st.columns(2)
with left_column:
    inpu_sector = st.radio(
        'What sector is your loan?',
        np.unique(kiva_loans['sector']))

with right_column:
    inpu_gender = st.radio(
        'What is your gender?',
        np.unique(kiva_loans['gender_bin']))

input_payment = st.slider('How much can you afford per month (USD)', 1.25, max(kiva_loans["repayment_per_mo"]), 1.0)
input_income = st.slider('What is your Country Income Index?', 0.36, max(kiva_loans["income_index"]), 0.1)

if st.button('Make Prediction'):
    input_sector = label_encoder_sector.transform(np.expand_dims(inpu_sector, -1))
    input_gender = label_encoder_gender.transform(np.expand_dims(inpu_gender, -1))
    inputs = np.expand_dims(
        [int(input_sector), int(input_gender), input_payment, input_income], 0)
    prediction = cat_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your recomended loan amount to request is: {np.squeeze(prediction, -1):.2f}g")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")

    