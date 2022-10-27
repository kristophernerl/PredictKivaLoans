# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 19:40:11 2022

@author: krist
"""
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import numpy as np
st.header("Kiva Loans Prediction App")
st.text_input("Enter your Name: ", key="name")
kiva_loans = pd.read_csv("https://raw.githubusercontent.com/kristophernerl/PredictKivaLoans/main/kiva_loans2.csv")

#load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model
rf_model = RandomForestRegressor(min_samples_split=2, min_samples_leaf=2, n_estimators =900)

##start
if st.checkbox('Show Training Dataframe'):
    kiva_loans

st.subheader("Please select relevant features of the loan you are looking for")
left_column, right_column = st.columns(2)
with left_column:
    inpu_sector = st.radio(
        'What sector is your loan:',
        np.unique(kiva_loans['sector']))


input_payment = st.slider('How much can you afford per month (USD)', 1.25, max(kiva_loans["repayment_per_mo"]), 71.87)
input_income = st.slider('What is your Country Income Index?', 0.36, max(kiva_loans["income_index"]), 0.832)

if st.button('Make Prediction'):
    input_sector = encoder.transform(np.expand_dims(inpu_sector, -1))
    inputs = np.expand_dims(
        [int(input_sector), input_payment, input_income], 0)
    prediction = rf_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your recomended loan amount to request is: {np.squeeze(prediction, -1):.2f}g")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")

    