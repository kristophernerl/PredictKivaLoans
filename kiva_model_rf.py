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

#load model from pickle file
import joblib
model = joblib.load("rfmodel.pkl")

#Allowing users to see the training dataframe if desired
if st.checkbox('Show Training Dataframe'):
    kiva_loans

st.subheader("Please select relevant features of the loan you are looking for")

#Creating Input Country Dropdown
country = (
        'Armenia',
        'Bolivia',
        'Cambodia',
        'Cameroon',
        'Colombia',
        'Ecuador',
        'Egypt',
        'El Salvador',
        'Ghana',
        'Guatemala',
        'Haiti',
        'Honduras',
        'India',
        'Indonesia',
        'Jordan',
        'Kenya',
        'Kyrgyzstan',
        'Lebanon',
        'Liberia',
        'Madagascar',
        'Mali',
        'Mexico',
        'Mozambique',
        'Nicaragua',
        'Nigeria',
        'Pakistan',
        'Paraguay',
        'Peru',
        'Philippines',
        'Samoa',
        'Sierra Leone',
        'Tajikistan',
        'Tanzania',
        'Timor-Leste',
        'Togo',
        'Turkey',
        'Uganda',
        'Vietnam',
        'Zimbabwe'
        )

input_country = st.selectbox("Select your Country", country)

#coding inputted Country to Income Index
input_income = 0.36
if input_country == 'Armenia':
   input_income =0.681
elif input_country == 'Bolivia':
   input_income =0.634
elif input_country == 'Cambodia':
   input_income =0.526
elif input_country == 'Cameroon':
   input_income =0.526
elif input_country == 'Colombia':
   input_income =0.735
elif input_country == 'Ecuador':
   input_income =0.699
elif input_country == 'Egypt':
   input_income =0.703
elif input_country == 'El Salvador':
   input_income =0.636
if input_country == 'Ghana':
   input_income =0.555
elif input_country == 'Guatemala':
   input_income =0.648
elif input_country == 'Haiti':
   input_income =0.425
elif input_country == 'Honduras':
   input_income =0.563
elif input_country == 'India':
   input_income =0.629
elif input_country == 'Indonesia':
   input_income =0.702
elif input_country == 'Jordan':
   input_income =0.667
elif input_country == 'Kenya':
   input_income =0.511
elif input_country == 'Kyrgyzstan':
   input_income =0.519
elif input_country == 'Lebanon':
   input_income =0.717
elif input_country == 'Liberia':
   input_income =0.36
elif input_country == 'Madagascar':
   input_income =0.396
elif input_country == 'Mali':
   input_income =0.442
elif input_country == 'Mexico':
   input_income =0.78
elif input_country == 'Mozambique':
   input_income =0.367
elif input_country == 'Nicaragua':
   input_income =0.592
elif input_country == 'Nigeria':
   input_income =0.597
elif input_country == 'Pakistan':
   input_income =0.592
elif input_country == 'Paraguay':
   input_income =0.716
elif input_country == 'Peru':
   input_income =0.723
elif input_country == 'Philippines':
   input_income =0.682
elif input_country == 'Samoa':
   input_income =0.615
elif input_country == 'Sierra Leone':
   input_income =0.394
elif input_country == 'Tajikistan':
   input_income =0.526
elif input_country == 'Tanzania':
   input_income =0.5
elif input_country == 'Timor-Leste':
   input_income =0.667
elif input_country == 'Togo':
   input_income =0.415
elif input_country == 'Turkey':
   input_income =0.824
elif input_country == 'Uganda':
   input_income =0.43
elif input_country == 'Vietnam':
   input_income =0.616
elif input_country == 'Zimbabwe':
   input_income =0.475

#creating Input Sector Radio Buttons
left_column, right_column = st.columns(2)
with left_column:
    inpu_sector = st.radio(
        'What sector is your loan?',
        np.unique(kiva_loans['sector']))

#creating Input Gender Radio Buttons
with right_column:
    inpu_gender = st.radio(
        'What is your gender?',
        np.unique(kiva_loans['gender_bin']))

#creating Input Monthly Payment Slider
input_payment = st.slider('How much can you afford per month (USD)', 1.25, max(kiva_loans["repayment_per_mo"]), 1.0)

#creating Prediction (button)
if st.button('Make Prediction'):
    input_sector = label_encoder_sector.transform(np.expand_dims(inpu_sector, -1))
    input_gender = label_encoder_gender.transform(np.expand_dims(inpu_gender, -1))
    inputs = np.expand_dims(
        [int(input_sector), int(input_gender), input_payment, input_income], 0)
    prediction = model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your recomended loan amount to request is: ${np.squeeze(prediction, -1):.2f}")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")

    